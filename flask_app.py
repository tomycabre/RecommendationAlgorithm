from flask import Flask, jsonify, abort, request
from recommender import init_model, get_recommendations
import sqlite3
import os

app = Flask(__name__)

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT id_lector FROM lectores")
user_ids = set(row[0] for row in cursor.fetchall())
conn.close()

@app.route("/api/version", methods=["GET"])
def version():
    return jsonify({"version": 2})

@app.route("/api/init", methods=["GET"])
def init():
    try:
        init_model()
    except Exception as e:
        abort(500, description=f"Error durante la inicialización: {str(e)}")
    return jsonify({"status": "ok"})

@app.route('/api/recomendar/<string:id_lector>', methods=['GET'])
def recomendar(id_lector):
    if id_lector not in user_ids:
        abort(404, description=f"User '{id_lector}' does not exist.")
    try:
        recommendations = get_recommendations(id_lector)
        recommended_items = [item_id for item_id, _ in recommendations]
        return jsonify({"recomendacion": recommended_items})
    except RuntimeError as e:
        abort(500, description=str(e))

@app.route('/api/recomendaciones', methods=['GET'])
def recomendaciones():
    try:
        data = request.get_json()
        id_lectores = data.get("id_lectores", [])
        if not id_lectores:
            abort(400, description="No se proporcionaron usuarios en la solicitud.")
        invalid_users = [id_lector for id_lector in id_lectores if id_lector not in user_ids]
        if invalid_users:
            abort(404, description=f"Usuarios no encontrados: {', '.join(invalid_users)}")
        all_recommendations = []
        for id_lector in id_lectores:
            recommendations = get_recommendations(id_lector)
            recommended_items = [item_id for item_id, _ in recommendations]
            all_recommendations.append({
                "id_lector": id_lector,
                "recomendacion": recommended_items
            })
        return jsonify({"recomendaciones": all_recommendations})
    except RuntimeError as e:
        abort(500, description=str(e))
    except Exception as e:
        abort(500, description=f"Error al procesar la solicitud: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)