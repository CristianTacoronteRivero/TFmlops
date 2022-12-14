"""Librería encargada de construir un pipeline para el dataset breast_cancer que se encuentra
en el modulo datasets de scikit-learn

Referencia: https://scikit-learn.org/stable/datasets/toy_dataset.html_
"""

import sklearn
from sklearn import datasets
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from types import NoneType
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error, max_error
import pickle


class BcSklearn:
    def __init__(self) -> None:
        self.X, self.y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

        self.get_X_y()

    def get_X_y(self) -> DataFrame | Series:
        """Devuelve dos variables donde uno son los features
         del dataset breast_cancer y el target

        :return: Features como objeto Dataframe y target como Series
        :rtype: DataFrame | Series
        """
        return self.X, self.y

    def split_train_test(
        self,
        X: DataFrame = None,
        y: Series = None,
        random_state: int = 42,
        test_size: float = 0.3,
    ) -> DataFrame | Series:
        """Método que se encarga de dividir los datos en entrenamiento y test

        :param X: Features, defaults to None
        :type X: DataFrame, optional
        :param y: Target, defaults to None
        :type y: Series, optional
        :param random_state:  defaults to 42
        :type random_state: int, optional
        :param test_size: Proporcion de datos para el proceso de test siendo el valorminimo 0 y maximo 1, defaults to 0.3
        :type test_size: float, optional
        :return: Devuelve 4 variables donde se obtiene los features y target tanto parael entrenamiento como para el test
        :rtype: DataFrame | Series
        """
        if isinstance(X, NoneType):
            X = self.X
        if isinstance(y, NoneType):
            y = self.y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def build_pipeline(
        self, X_train: DataFrame
    ) -> sklearn.ensemble._voting.VotingRegressor:
        """Método encargado de construir un modelo simple mediante el método ensamblado
        "voting"

        :param X_train: Dataframe de "muestra" para especificar a qué columnas se ledeben de realizar las trasnformaciones
        :type X_train: DataFrame
        :return: Modelo no ajustado pero completamente construido
        :rtype: sklearn.ensemble._voting.VotingRegressor
        """
        pipe_num = make_pipeline(KNNImputer(), StandardScaler(), PCA())
        column_transformer = ColumnTransformer(
            [("pipe_num", pipe_num, list(X_train.select_dtypes("number").columns))]
        )
        model_build = make_pipeline(
            column_transformer,
            VotingRegressor(
                [("KNN", KNeighborsRegressor()), ("Forest", RandomForestRegressor())]
            ),
        )

        return model_build

    def search_and_fit(
        self,
        model: sklearn.ensemble._voting.VotingRegressor,
        X: DataFrame,
        y: Series,
        param_model_knn: dict,
        param_model_forest: dict,
        cv: int = 5,
    ) -> sklearn.model_selection._search.GridSearchCV:
        """Método encargado de realizar una búsqueda de los mejores hiperparámetros
        del modelo

        :param model: Modelo encargado de realizar las predicciones
        :type model: sklearn.ensemble._voting.VotingRegressor
        :param X: Features
        :type X: DataFrame
        :param y: Target
        :type y: Series
        :param param_model_knn: Diccionario de parámetros a modificar para el modelo KNN
        :type param_model_knn: dict(str,list)
        :param param_model_forest: Diccionario de parámetros a modificar para el modelo RandomForest
        :type param_model_forest: dict(str,list)
        :param cv: Folds del proceso Cross Validation, defaults to 5
        :type cv: int, optional
        :return: Modelo óptimo
        :rtype: sklearn.model_selection._search.GridSearchCV
        """
        param_raiz = {
            "votingregressor__KNN__" + k: v for k, v in param_model_knn.items()
        }
        param_forest = {
            "votingregressor__Forest__" + k: v for k, v in param_model_forest.items()
        }
        param_raiz.update(param_forest)

        grid = GridSearchCV(
            estimator=model, param_grid=param_raiz, cv=cv, verbose=1
        ).fit(X, y)

        self.grid = grid

        return grid

    def metricas_regresion(self, X_test: DataFrame, y_test: Series) -> float:
        """Método que devuelve los parámetros r2, mse y me

        :param X_test: Features para testear el modelo
        :type X_test: DataFrame
        :param y_test: Target para testear el modelo
        :type y_test: Series
        :return: Valores que representa la calidad del modelo
        :rtype: float
        """
        r2 = r2_score(y_test, self.grid.predict(X_test))
        mse = mean_squared_error(y_test, self.grid.predict(X_test))
        # me = max_error(y_test, self.grid.predict(X_test))

        return r2, mse

    def save_model(self, name: str):
        """Método para generar un archivo .pkl que
        guarda el modelo entrenado

        :param name: Nombre del fichero .pkl
        :type name: str
        """
        with open(name, "wb") as f:
            pickle.dump(name, f)
