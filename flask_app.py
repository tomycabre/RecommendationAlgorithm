from flask import Flask, jsonify, abort
from recommender import get_recommendations
import sqlite3
import os

app = Flask(__name__)

# Load user IDs into memory at startup
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT id_lector FROM lectores")
user_ids = set(row[0] for row in cursor.fetchall())
conn.close()

@app.route("/api/version", methods=["GET"])
def version():
    return jsonify({"version": 2})

@app.route('/api/recomendar/<string:id_lector>', methods=['GET'])
def recomendar(id_lector):
    if id_lector not in user_ids:
        # User does not exist; return 404 error
        abort(404, description=f"User '{id_lector}' does not exist.")

    # Proceed to get recommendations
    recommendations = get_recommendations(id_lector)
    # Extract only the item_ids from recommendations
    recommended_items = [item_id for item_id, _ in recommendations]
    response = {"recomendacion": recommended_items}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)