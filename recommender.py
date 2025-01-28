import os
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import AlgoBase, PredictionImpossible

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "data.db")

global_model = None
book_ids = None

def load_book_ids():
    global book_ids
    if book_ids is None:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id_libro FROM libros")
        book_ids = set(row[0] for row in cursor.fetchall())
        conn.close()
    return book_ids

class HybridRecommender(AlgoBase):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, trainset):
        super().fit(trainset)
        self.cf_algo = SVD(n_factors=50, n_epochs=15, lr_all=0.005, reg_all=0.02)
        self.cf_algo.fit(trainset)
        self.global_mean = trainset.global_mean

        with sqlite3.connect(db_path) as conn:
            self.item_content = pd.read_sql_query(
                "SELECT id_libro, autor, genero FROM libros", conn
            ).set_index('id_libro')

            # Popularidad por rating promedio
            pop_df = pd.read_sql_query(
                "SELECT id_libro, AVG(rating) as avg_rating FROM interacciones GROUP BY id_libro",
                conn
            )
        self.item_pop = dict(zip(pop_df['id_libro'].astype(str), pop_df['avg_rating']))
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.global_mean

        uid = self.trainset.to_raw_uid(u)
        iid = self.trainset.to_raw_iid(i)
        try:
            cf_est = self.cf_algo.predict(uid, iid, verbose=False).est
        except PredictionImpossible:
            cf_est = self.global_mean

        # Contenido
        user_ratings = self.trainset.ur[u]
        sim_sum, sim_weighted_ratings = 0, 0
        for j, r in user_ratings:
            iid_j = self.trainset.to_raw_iid(j)
            sim = self.content_similarity(iid, iid_j)
            sim_sum += sim
            sim_weighted_ratings += sim * r
        content_est = (sim_weighted_ratings / sim_sum) if sim_sum > 0 else self.global_mean

        # Popularidad
        pop_est = self.item_pop.get(iid, self.global_mean) / 10.0

        return (self.alpha * cf_est) + (self.beta * content_est) + (self.gamma * pop_est)

    def content_similarity(self, iid1, iid2):
        if (iid1 not in self.item_content.index) or (iid2 not in self.item_content.index):
            return 0
        item1 = self.item_content.loc[iid1]
        item2 = self.item_content.loc[iid2]
        score = 0
        if item1['autor'] == item2['autor']:
            score += 0.5
        if item1['genero'] == item2['genero']:
            score += 0.5
        return score

def load_data():
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query(
            "SELECT id_lector AS user_id, id_libro AS item_id, rating FROM interacciones", conn
        )
    reader = Reader(rating_scale=(1, 10))
    return Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

def train_model(trainset):
    algo = HybridRecommender()
    algo.fit(trainset)
    return algo

def init_model():
    global global_model
    data = load_data()
    trainset = data.build_full_trainset()
    global_model = train_model(trainset)

def get_recommendations(user_id, n=10):
    if global_model is None:
        raise RuntimeError("Modelo no inicializado.")
    with sqlite3.connect(db_path) as conn:
        rated_items_df = pd.read_sql_query(
            "SELECT id_libro FROM interacciones WHERE id_lector = ?",
            conn, params=(user_id,)
        )
        rated_items = set(rated_items_df['id_libro'].astype(str))
        candidate_items_df = pd.read_sql_query(
            """
            SELECT DISTINCT id_libro
            FROM interacciones
            WHERE id_libro NOT IN ({})
            LIMIT 500
            """.format(",".join(["?"]*len(rated_items))) if rated_items else
            "SELECT DISTINCT id_libro FROM interacciones LIMIT 500",
            conn,
            params=tuple(rated_items) if rated_items else ()
        )

    candidate_ids = candidate_items_df['id_libro'].astype(str).tolist()
    predictions = []
    for cid in candidate_ids:
        pred = global_model.predict(str(user_id), cid, verbose=False)
        predictions.append((cid, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]