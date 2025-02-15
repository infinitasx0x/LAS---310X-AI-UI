from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import logging
import os
import subprocess

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


n_threads = os.cpu_count() // 2  

model_path = r"模型存放地址"
try:
    llm = Llama(model_path=model_path, n_threads=n_threads, n_ctx=4096)
    logging.info(f"✅ Model loaded successfully with {n_threads} threads.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    exit(1)  

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400


        final_prompt = (
            "You are a helpful AI assistant.\n"
            "Answer questions concisely and accurately.\n\n"
            f"User: {user_input}\n\n"
            "AI:"
        )
        logging.debug(f"Formatted prompt: {final_prompt}")


        try:
            response = llm.create_completion(
                prompt=final_prompt,
                max_tokens=200,      
                temperature=0.2,      
                top_p=0.85,          
                repeat_penalty=2.0,   
                top_k=300,            
                stop=["\nUser:", "User:"]  
            )
            logging.debug(f"Full response: {response}")
        except RuntimeError as e:
            logging.error(f"RuntimeError in LLM call: {e}")
            return jsonify({"error": "Model error"}), 500
        except Exception as e:
            logging.error(f"Unexpected error in LLM call: {e}")
            return jsonify({"error": "An unexpected error occurred"}), 500

  
        choices = response.get("choices", [])
        if choices:
            ai_response = choices[0].get("text", "").strip()
            logging.debug(f"AI response: {ai_response}")
            return jsonify({"response": ai_response})
        else:
            logging.error("❌ No choices in response")
            return jsonify({"error": "No response generated"}), 500

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
