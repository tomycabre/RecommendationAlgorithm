import pandas as pd
import sqlite3
import os
from surprise import Dataset, Reader, SVD
from surprise import AlgoBase, PredictionImpossible

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "data.db")

# Variable global para almacenar el modelo entrenado
global_model = None  # Se inicializa en None y se asigna en /api/init

# Variable global para almacenar los IDs de los libros
book_ids = None  # Inicializamos la variable como None

def load_book_ids():
    """
    Carga los IDs de los libros desde la base de datos una sola vez.
    """
    global book_ids
    if book_ids is None:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id_libro FROM libros")
        book_ids = set(row[0] for row in cursor.fetchall())  # Guardamos los IDs en una variable global
        conn.close()
    return book_ids

class CollaborativeRecommender(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Entrenar modelo de filtrado colaborativo
        self.cf_algo = SVD()
        self.cf_algo.fit(trainset)
        # Calcular rating medio global
        self.global_mean = trainset.global_mean
        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            uid = self.trainset.to_raw_uid(u)
            iid = self.trainset.to_raw_iid(i)

            # Predicción de filtrado colaborativo
            try:
                cf_est = self.cf_algo.predict(uid, iid, verbose=False).est
            except PredictionImpossible:
                cf_est = self.global_mean
        else:
            # Retorna rating medio global para usuario o item desconocido
            cf_est = self.global_mean

        return cf_est

def load_data():
    conn = sqlite3.connect(db_path)
    query = "SELECT id_lector AS user_id, id_libro AS item_id, rating FROM interacciones"
    data = pd.read_sql_query(query, conn)
    conn.close()
    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    return dataset

def train_model(trainset):
    """
    Entrena el modelo y devuelve el objeto entrenado.
    """
    algo = CollaborativeRecommender()
    algo.fit(trainset)
    return algo

def init_model():
    """
    Inicializa el modelo global cargando datos y entrenándolo.
    """
    global global_model
    data = load_data()
    trainset = data.build_full_trainset()
    global_model = train_model(trainset)
    print("Modelo global inicializado correctamente.")

def get_recommendations(user_id, n=10):
    """
    Devuelve las mejores n recomendaciones para el usuario dado.
    """
    if global_model is None:
        raise RuntimeError("El modelo no está inicializado. Llame a /api/init primero.")

    # Cargar los IDs de los libros solo si es necesario
    book_ids = load_book_ids()

    # Abrir conexión para consultas adicionales
    conn = sqlite3.connect(db_path)

    # Obtener items valorados por el usuario
    rated_items_df = pd.read_sql_query(
        "SELECT id_libro FROM interacciones WHERE id_lector = ?",
        conn, params=(user_id,)
    )
    rated_items = set(rated_items_df['id_libro'].astype(str).tolist())

    # Encontrar usuarios con al menos 4 libros valorados en común
    common_users_df = pd.read_sql_query(
        """
        SELECT id_lector
        FROM interacciones
        WHERE id_libro IN ({})
        GROUP BY id_lector
        HAVING COUNT(DISTINCT id_libro) >= 4
        """.format(",".join("?" * len(rated_items))),
        conn, params=tuple(rated_items)
    )
    common_users = set(common_users_df['id_lector'].tolist())

    # Obtener libros valorados por esos usuarios, que el usuario actual no ha valorado
    candidate_items_df = pd.read_sql_query(
        """
        SELECT DISTINCT id_libro
        FROM interacciones
        WHERE id_lector IN ({}) AND id_libro NOT IN ({})
        """.format(",".join("?" * len(common_users)), ",".join("?" * len(rated_items))),
        conn, params=tuple(common_users) + tuple(rated_items)
    )
    candidate_items = candidate_items_df['id_libro'].astype(str).tolist()

    conn.close()

    # Predecir ratings para los libros candidatos
    predictions = []
    for item_id in candidate_items:
        pred = global_model.predict(user_id, item_id, verbose=False)
        predictions.append((item_id, pred.est))

    # Obtener las top N recomendaciones
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    return top_n
