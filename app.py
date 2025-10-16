import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU to avoid DLL issues

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pymysql

app = Flask(__name__)

# ---- Database connection setup ----
$servername = "localhost"; 
$dbUsername = "u120901047_Ay33U"; 
$dbPassword = "JJms@1010";        
$dbname     = "u120901047_7p38k";   
$conn = new mysqli();

# Create connection function
def get_db_connection():
    return pymysql.connect(
        host=$servername,
        user=$dbUsername,
        password=$dbPassword,
        database=$dbname,
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

