import pandas as pd
import sqlite3
import os
from surprise import Dataset, Reader, SVD
from surprise import AlgoBase, PredictionImpossible

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "data.db")
print(f"Database path: {db_path}")  # Para depuración

class HybridRecommender(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Entrenar modelo de filtrado colaborativo
        self.cf_algo = SVD()
        self.cf_algo.fit(trainset)
        # Calcular rating medio global
        self.global_mean = trainset.global_mean
        # Cargar datos de contenido de items
        conn = sqlite3.connect(db_path)
        self.item_content = pd.read_sql_query(
            "SELECT id_libro, autor, genero FROM libros", conn
        ).set_index('id_libro')
        conn.close()
        return self

    def estimate(self, u, i):
        # Verificar si el usuario y el item son conocidos
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

            # Estimación basada en contenido
            user_ratings = self.trainset.ur[u]
            sim_sum = 0
            sim_weighted_ratings = 0
            for j, r in user_ratings:
                iid_j = self.trainset.to_raw_iid(j)
                try:
                    sim = self.content_similarity(iid, iid_j)
                    sim_sum += sim
                    sim_weighted_ratings += sim * r
                except KeyError:
                    continue
            if sim_sum > 0:
                content_est = sim_weighted_ratings / sim_sum
            else:
                content_est = self.global_mean
            # Estimación híbrida
            est = (cf_est + content_est) / 2
        else:
            # Retorna rating medio global para usuario o item desconocido
            est = self.global_mean
        return est

    def content_similarity(self, iid1, iid2):
        item1 = self.item_content.loc[iid1]
        item2 = self.item_content.loc[iid2]
        sim = 0
        if item1['autor'] == item2['autor']:
            sim += 0.5
        if item1['genero'] == item2['genero']:
            sim += 0.5
        return sim

def list_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:", tables)
    conn.close()

list_tables()  # Para depuración

def load_data():
    conn = sqlite3.connect(db_path)
    query = "SELECT id_lector AS user_id, id_libro AS item_id, rating FROM interacciones"
    data = pd.read_sql_query(query, conn)
    conn.close()
    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    return dataset

def train_model(trainset):
    algo = HybridRecommender()
    algo.fit(trainset)
    return algo

def get_recommendations(user_id, n=5):
    # Cargar datos y entrenar el modelo
    data = load_data()
    trainset = data.build_full_trainset()
    algo = train_model(trainset)

    # Abrir conexión para consultas adicionales
    conn = sqlite3.connect(db_path)

    # Obtener items ya valorados por el usuario
    rated_items_df = pd.read_sql_query(
        "SELECT id_libro FROM interacciones WHERE id_lector = ?",
        conn, params=(user_id,)
    )
    rated_items = rated_items_df['id_libro'].astype(str).tolist()

    # Obtener todos los IDs de items
    all_items_df = pd.read_sql_query("SELECT id_libro FROM libros", conn)
    all_items = all_items_df['id_libro'].astype(str).tolist()
    conn.close()

    # Predecir ratings para items no valorados
    unrated_items = [item for item in all_items if item not in rated_items]
    predictions = []
    for item_id in unrated_items:
        pred = algo.predict(user_id, item_id, verbose=False)
        predictions.append((item_id, pred.est))

    # Obtener las top N recomendaciones
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    return top_n