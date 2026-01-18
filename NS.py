import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from abc import ABC, abstractmethod
import numpy as np
import os
from rich.console import Console
import gpflow as gpf
import tensorflow as tf

console = Console()
os.environ["GPFLOW_FLOAT"] = 'float64'

os.environ["TF_USE_LEGACY_KERAS"] = 'True'


class Model(ABC):
    def __init__(self, x_names, y_name, *args, **kwargs):
        """Generic Z-score calculator
        Parameters
        ----------
        x_names : list of str
            Names of the input columns (covariates)
        y_name : str
            Name of the target (y) column
        """
        self.x_names = x_names
        self.y_name = y_name

    @abstractmethod
    def fit(self, df_train):
        """Fit model parameters on data

        Parameters
        ----------
        df_train : dataframe
            Training set

        """
        pass

    
def _desc(X, y, model):
    import tensorflow as tf

    def ps(data, is_Z=False):
        ll = np.percentile(data,10)
        lo = np.percentile(data,25)
        m = np.percentile(data,50)
        h = np.percentile(data,75)
        hh = np.percentile(data,90)
        s = f'({ll:.1f},{lo:.1f},{m:.1f},{h:.1f},{hh:.1f})'
        if is_Z:
            num_anomal = int(np.sum(data > 2))
            s += f'[{num_anomal:3d}]'
        return s
    Ymean, Yvar = model.predict_y(X)
    Ymean = Ymean.numpy().squeeze()
    Ystd = tf.sqrt(Yvar).numpy().squeeze()
    Z = (y-Ymean[:,None])/Ystd[:,None]
    ret = f'{ps(Ymean)}±{ps(Ystd)}'
    ret = ret + f' |{np.mean(np.abs(Ymean[:,None] - y)):.1f}|'
    ret = ret + f' Z={ps(Z, is_Z=True)}'
    return ret


class HGPRModel(Model):

    class ConstantHeteroscedasticPrior(gpf.functions.Constant, gpf.functions.Function):
        import tensorflow as tf
        from check_shapes import inherit_check_shapes

        def __init__(self, mean_c, var_c, output_dim: int = 2):
            gpf.functions.Constant.__init__(self)
            self.output_dim = output_dim
            del self.c
            self.mean_c = mean_c
            self.var_c = var_c

        # @inherit_check_shapes
        def __call__(self, X):
            # output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
            return tf.constant([self.mean_c, self.var_c], dtype='float64')

    class Hyper():
        def __init__(self,
                     kernel='Matern12',
                     mean_c=0,
                     var_c=0,
                     noise_1=0,
                     noise_2=0,
                     num_inducing_variables=10,
                     natgrad_gamma=0.02,
                     adam_learning_rate=0.01,
                     epochs=40000,
                     *args, **kwargs):
            self.mean_c = mean_c
            self.var_c = var_c
            self.kernel = kernel
            self.noise_1 = noise_1
            self.noise_2 = noise_2
            self.num_inducing_variables = num_inducing_variables
            self.natgrad_gamma = natgrad_gamma
            self.adam_learning_rate = adam_learning_rate
            self.epochs = epochs

        def __str__(self):
            return str(self.__dict__)

        def __repr__(self):
            return str(self.__dict__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded_from_file = False
        self.hyper = self.Hyper(*args, **kwargs)

    def fit(self, df_train):
        import tensorflow as tf
        import tensorflow_probability as tfp
        import gpflow as gpf
        import hashlib
        # If model was already trained on the same data, predictors, and hyperparameters, load from file
        hasher = hashlib.blake2b(digest_size=20)
        signature = f'{str(self.x_names)}\n'
        signature += f'{str(self.y_name)}\n'
        signature += f'{str(self.hyper)}\n'
        signature += '\n'.join([str(a) for a in df_train.index])
        hasher.update(str.encode(signature))
        save_dir = os.path.join('gpflow_checkpoints', hasher.hexdigest())
        if os.path.exists(save_dir):
            console.print(f'[green]Loading trained model from {save_dir}[/]')
            self.model = tf.saved_model.load(save_dir)
            with open(os.path.join(save_dir, 'hyper'), 'r') as istream:
                hyper = eval(istream.readline().strip())
                self.model.hyper = self.Hyper(**hyper)
            self.loaded_from_file = True
            return

        np.random.seed(0)
        tf.random.set_seed(0)
        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
            # distribution_class=lambda loc, scale: tfp.distributions.StudentT(3.0, loc, scale),
            scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
            # scale_transform=tfp.bijectors.Softplus(),  # Softplus Transform
        )
        X_train = df_train[self.x_names].values.astype(np.float64)
        y_train = df_train[self.y_name].values.astype(np.float64)[:]
        D = X_train.shape[1]
        console.print(f'{X_train.shape=}, {y_train.shape=}')

        # kernels
        k1 = eval(f'gpf.kernels.{self.hyper.kernel}(lengthscales=np.ones(D).astype(np.float64))')
        k2 = eval(f'gpf.kernels.{self.hyper.kernel}(lengthscales=np.ones(D).astype(np.float64))')
        if self.hyper.noise_1 > 0:
            k1 = k1 + gpf.kernels.White(variance=self.hyper.noise_1)
        if self.hyper.noise_2 > 0:
            k2 = k2 + gpf.kernels.White(variance=self.hyper.noise_2)
        kernel = gpf.kernels.SeparateIndependent([k1, k2])

        # Initialize inducing locations to the first M inputs in the dataset
        Z1 = X_train[:self.hyper.num_inducing_variables, :].copy()
        Z2 = X_train[:self.hyper.num_inducing_variables, :].copy()
        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(Z1),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(Z2),  # This is U2 = f2(Z2)
            ]
        )
        self.model = gpf.models.SVGP(
            # mean_function=gpf.functions.Identity(),
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
            num_data=X_train.shape[0],
            mean_function=HGPRModel.ConstantHeteroscedasticPrior(self.hyper.mean_c, self.hyper.var_c),
        )

        console.print(f'[yellow]Training HGPR {self.x_names}->{self.y_name} with {self.hyper}[/]')
        data = (X_train, y_train)
        loss_fn = self.model.training_loss_closure(data)
        gpf.utilities.set_trainable(self.model.q_mu, False)
        gpf.utilities.set_trainable(self.model.q_sqrt, False)
        gpf.set_trainable(self.model.inducing_variable, False)

        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=self.hyper.natgrad_gamma)
        adam_vars = self.model.trainable_variables
        adam_opt = tf.optimizers.legacy.Adam(learning_rate=self.hyper.adam_learning_rate)

        @tf.function
        def optimisation_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        log_freq = 2000
        # import pdb; pdb.set_trace()
        for epoch in range(1, self.hyper.epochs + 1):
            optimisation_step()
            if epoch % log_freq == 0 and epoch > 0:
                loss_s = f'{epoch:5d} L={loss_fn().numpy():4.1f}'
                console.print(f'{loss_s} [green]T: {_desc(X_train,y_train, self.model)}[/]')

        # Save the trained model in a dir named from the hash of the signature string
        self.model.compiled_predict_f = tf.function(
            lambda Xnew: self.model.predict_f(Xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, D], dtype=tf.float64)],
        )
        self.model.compiled_predict_y = tf.function(
            lambda Xnew: self.model.predict_y(Xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, D], dtype=tf.float64)],
        )
        tf.saved_model.save(self.model, save_dir)
        with open(os.path.join(save_dir,'signature'), 'w') as ostream:
            print(signature, file=ostream)
        # Also save hyperparameters in the same directory
        with open(os.path.join(save_dir,'hyper'), 'w') as ostream:
            print(self.hyper, file=ostream)

    def predict_y(self, df_test):
        if self.loaded_from_file:
            return self.model.compiled_predict_y(df_test[self.x_names].values.astype(np.float64))
        else:
            return self.model.predict_y(df_test[self.x_names].values.astype(np.float64))

    def predict_f(self, df_test):
        if self.loaded_from_file:
            return self.model.compiled_predict_f(df_test[self.x_names].values.astype(np.float64))
        else:
            return self.model.predict_f(df_test[self.x_names].values.astype(np.float64))



def NormalcyScore(RawDataSet, MyColList, MyContextList, MyBehaveList, sample_value):
    
    MyDataSet = RawDataSet[MyColList].copy()
    MyContextDataSet = MyDataSet[MyContextList]
    MyBehaveDataSet = MyDataSet[MyBehaveList]

    anomaly_scores = np.zeros(len(MyDataSet))
    hdis = np.zeros(len(MyDataSet))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(MyDataSet):
        X_train, X_test = MyContextDataSet.iloc[train_index], MyContextDataSet.iloc[test_index]
        y_train, y_test = MyBehaveDataSet.iloc[train_index], MyBehaveDataSet.iloc[test_index]

        # Instantiate and train the HGPR model
        hgpr_model = HGPRModel(
            x_names=MyContextList,
            y_name=MyBehaveList,
            mean_c=y_test[MyBehaveList[0]].mean(),
            var_c=-5,
            kernel="RationalQuadratic",
            # kernel="RBF",
            # kernel="Matern12",
            num_inducing_variables=len(X_train)//20,
            epochs=40)
        
        
        hgpr_model.fit(pd.concat([X_train, y_train], axis=1))

        class Res:
            pass
        res = Res()
        
        m, s = hgpr_model.predict_f(X_test)
        res.m_1 = m[:, 0].numpy().squeeze() 
        res.s_1 = s[:, 0].numpy().squeeze() 
        res.m_2 = m[:, 1].numpy().squeeze() 
        res.s_2 = s[:, 1].numpy().squeeze() 

        for idx, test_idx in enumerate(test_index):
            observed_value = y_test.iloc[idx][MyBehaveList[0]]
            
            z_score = (observed_value - res.m_1[idx]) * np.exp(-res.m_2[idx] + res.s_2[idx] ** 2 / 2)
            
            # HDI interval
            import arviz as az

            sample_size=30000
            f1 = res.m_1[idx] + res.s_1[idx] * np.random.normal(size=sample_size)
            f2 = res.m_2[idx] + res.s_2[idx] * np.random.normal(size=sample_size)

            z = (observed_value - f1) / np.exp(f2)

            hdi = az.hdi(z, hdi_prob=0.95)

            hdi_len = np.abs(hdi[1]-hdi[0])
            anomaly_scores[test_idx] = np.abs(z_score)
            hdis[test_idx] = hdi_len



    MyDataSet['anomaly_score'] = anomaly_scores

    my_roc_score = roc_auc_score(MyDataSet["ground_truth"], MyDataSet["anomaly_score"])

    TempDataSet = MyDataSet[["ground_truth", "anomaly_score"]]
    P_TempDataSet = TempDataSet.sort_values(by=['anomaly_score'], ascending=[False]).head(sample_value)
    
    TP_value = (P_TempDataSet["ground_truth"] == 1).sum()
    P_at_n_value = TP_value / sample_value if sample_value > 0 else 0
    
    y = np.array(MyDataSet["ground_truth"])
    pred = np.array(MyDataSet["anomaly_score"])
    precision, recall, _ = precision_recall_curve(y, pred, pos_label=1)
    my_pr_auc = auc(recall, precision)

    return my_pr_auc, my_roc_score, P_at_n_value, MyDataSet
