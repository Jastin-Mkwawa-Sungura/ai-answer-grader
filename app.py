import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU to avoid DLL issues

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pymysql

app = Flask(__name__)

# ---- Database connection setup ----
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""         # <-- Change if your MySQL has a password
DB_NAME = "chat_validation"

# Create connection function
def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )

# ---- Load AI model ----
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- Define reference question and correct answer ----
QUESTION = "What is the capital of France?"
CORRECT_ANSWER = "The capital of France is Paris."

# ---- API endpoint for grading ----
@app.route('/grade', methods=['POST'])
def grade_answer():
    try:
        data = request.get_json()
        username = data.get("username", "anonymous")
        user_answer = data.get("answer", "")

        # Encode answers
        embedding1 = model.encode(user_answer, convert_to_tensor=True)
        embedding2 = model.encode(CORRECT_ANSWER, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.cos_sim(embedding1, embedding2).item()

        # Convert to percentage
        score = round(similarity * 100, 2)
        if score < 0: score = 0
        if score > 100: score = 100

        # Store in MySQL
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_answers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100),
                    question TEXT,
                    user_answer TEXT,
                    ai_score FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                INSERT INTO user_answers (username, question, user_answer, ai_score)
                VALUES (%s, %s, %s, %s)
            """, (username, QUESTION, user_answer, score))
            conn.commit()
        conn.close()

        return jsonify({
            "question": QUESTION,
            "user_answer": user_answer,
            "correct_answer": CORRECT_ANSWER,
            "score_percentage": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Run the app ----
if __name__ == '__main__':
    app.run(debug=True)
