from flask import Flask, render_template, request, jsonify, session
import csv
import os
import time
import secrets
import random
import openai
import json
import base64
from dotenv import load_dotenv
from webdav3.client import Client
from config import NUM_ANSWERS, CERTAINTY_TIME_SECONDS, QUESTION_TIME_SECONDS

# Load environment variables
load_dotenv()

# Initialize the API key and model name once at startup
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
    
LLM_MODEL = os.getenv('LLM_ENGINE', 'gpt-4.1-mini')

# Create the OpenAI client once
openai_client = openai.OpenAI(api_key=api_key)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["SESSION_PERMANENT"] = False

# Server-side cache for storing images and other large data
# This keeps data out of the session cookie
cache = {
    "images": {},  # Will store image data by task_id
    "tasks": None,  # Will store all tasks
}

# Pre-configure WebDAV client for faster connections
webdav_options = {
    'webdav_hostname': os.getenv('SCIEBO_URL'),
    'webdav_login': os.getenv('SCIEBO_LOGIN'),
    'webdav_password': os.getenv('SCIEBO_PASSWORD'),
    'connect_timeout': 10,
    'read_timeout': 30
}
webdav_client = Client(webdav_options)

class RecordKeeper:
    def __init__(self, session):
        self.session = session
        
    def start_task_record(self, task_idx, task, options):
        # Store only necessary data
        self.session["record_data"]["current_task"] = {
            "task_index": task_idx,
            "question": task["question"],
            "options": options,
            "solution": task["correct_solution"],
            "user_inputs": [],
            "chat_history": [],
            "time_spent": 0,
            "final_answer": None,
            "certainty": None
        }
        
    def add_user_input(self, input_text):
        if self.session["record_data"]["current_task"]:
            self.session["record_data"]["current_task"]["user_inputs"].append({
                "timestamp": time.time(),
                "input": input_text
            })
            
    def add_chat_interaction(self, user_msg, assistant_msg):
        if self.session["record_data"]["current_task"]:
            self.session["record_data"]["current_task"]["chat_history"].append({
                "timestamp": time.time(),
                "user": user_msg,
                "assistant": assistant_msg
            })
            
    def complete_task(self, answer, certainty, time_spent):
        if self.session["record_data"]["current_task"]:
            current = self.session["record_data"]["current_task"]
            current.update({
                "final_answer": answer,
                "certainty": certainty,
                "time_spent": time_spent
            })
            self.session["record_data"]["records"].append(current)
            self.session["record_data"]["current_task"] = None
            
    def save_and_send(self, stats):
        final_data = {
            "prolific_id": self.session["prolific_id"],
            "group": "intervention" if self.session["intervention_group"] else "control",
            "tasks": self.session["record_data"]["records"],
            "statistics": stats
        }
        
        try:
            filename = f'results_{self.session["prolific_id"]}.json'
            filepath = os.path.join("results", filename)
            
            os.makedirs("results", exist_ok=True)
            
            # Write file with minimal indent for faster processing
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, ensure_ascii=False, fp=f)

            # Use the pre-configured WebDAV client for faster upload
            remote_path = os.path.join(os.getenv('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
            
            webdav_client.upload_file(
                remote_path=remote_path,
                local_path=filepath
            )
            
            print(f"File successfully uploaded to Sciebo: {remote_path}")
            os.remove(filepath)
        except Exception as e:
            print(f"Error saving/uploading results: {str(e)}")

def load_tasks():
    """Load tasks once and cache them to avoid repeated file operations"""
    if cache["tasks"] is None:
        result = []
        csv_file = os.path.join("data", "tasks.csv")
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    task = {
                        "question": row["question"].strip(),
                        "options": [x.strip() for x in row["options"].split(";")],
                        "correct_solution": row["correct_solution"].strip(),
                        "image_path": row.get("image_path", "").strip()
                    }
                    if task["correct_solution"] not in task["options"]:
                        task["options"].append(task["correct_solution"])
                    result.append(task)
            cache["tasks"] = result
        except Exception as e:
            print(f"Error loading tasks: {e}")
            cache["tasks"] = []
    return cache["tasks"]

def prepare_options(task):
    """Prepare randomized options for a task"""
    options = task['options'].copy()
    correct = task['correct_solution']
    if correct in options:
        options.remove(correct)
    random.shuffle(options)
    selected = options[:NUM_ANSWERS-1]
    final = selected + [correct]
    random.shuffle(final)
    return final

def cache_image(task_idx, image_path):
    """Cache an image as base64 for a given task, replacing any previously cached image"""
    if not image_path:
        return None
        
    # Use a constant key for current task image
    cache_key = "current_task_image"
    
    img_path = os.path.join("static", "img", image_path + ".jpg")
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as img_file:
                image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Replace any existing cached image with the current one
                cache["images"] = {cache_key: base64_image}
                return cache_key
        except Exception as e:
            print(f"Error caching image: {str(e)}")
    
    return None

def get_cached_image(cache_key):
    """Get a cached image by key"""
    return cache["images"].get(cache_key)

class SessionManager:
    def __init__(self, session):
        self.session = session
        self.record_keeper = RecordKeeper(session)
    
    def init_state(self):
        defaults = {
            "phase": "prolific",
            "lines_left": [],
            "lines_right": [],
            "idx": 0,
            "results": [],
            "start_time": None,
            "prolific_id": None,
            "certainty_pending": False,
            "current_result": None,
            "intervention_group": random.choice([True, False]),
            "current_image_key": None,  # Store the key to cached image instead of the image itself
        }
        for key, value in defaults.items():
            if key not in self.session:
                self.session[key] = value
        
        if "record_data" not in self.session:
            self.session["record_data"] = {
                "records": [],
                "current_task": None
            }

    def clear_console(self):
        self.session["lines_left"] = []
        self.session["lines_right"] = []
    
    def append_left(self, txt):
        lines = self.session["lines_left"]
        lines.append(txt)
    
    def append_right(self, txt):
        lines = self.session["lines_right"]
        lines.append(txt)
    
    def reset(self):
        self.session.clear()
        self.init_state()

class QuestionManager:
    def __init__(self, session_manager):
        self.sm = session_manager
    
    def show_question(self):
        i = self.sm.session["idx"]
        tlist = load_tasks()  # Load from cache
        if i >= len(tlist):
            self.sm.session["phase"] = "summary"
            self.sm.clear_console()
            return self.show_summary()
        
        self.sm.session["start_time"] = time.time()
        t = tlist[i]
        
        # First display the question number
        self.sm.append_left(f"$  QUESTION {i+1}/{len(tlist)}\n")
        
        # Cache the image and store only the reference key in the session
        self.sm.session["current_image_key"] = None  # Reset previous image cache
        if t["image_path"]:
            img_path = os.path.join("static", "img", t["image_path"] + ".jpg")
            if os.path.exists(img_path):
                # Display the image in the UI
                self.sm.append_left(f"""<img src="static/img/{t['image_path']}.jpg" style="width:400px;">""")
                
                # Cache the image and store only the key in session
                self.sm.session["current_image_key"] = cache_image(i, t["image_path"])
        
        # Then display the question text
        self.sm.append_left(f"   {t['question']}")
        
        options = prepare_options(t)
        self.sm.session['current_options'] = options
        
        if "record_data" not in self.sm.session:
            self.sm.session["record_data"] = {
                "records": [],
                "current_task": None
            }
        self.sm.record_keeper.start_task_record(i, t, options)
        
        # Finally display the answer options
        for x, op in enumerate(options):
            self.sm.append_left(f"   {chr(ord('a') + x)}) {op}")
        
        self.sm.append_left("$  Type a letter and press ENTER.")
    
    def show_summary(self):
        rs = self.sm.session["results"]
        stats = self.calculate_stats(rs)
        self.sm.record_keeper.save_and_send(stats)
        
        # Add thank you message at the beginning
        self.sm.append_left("$  Thank you for participating in our study!")
        self.sm.append_left("$  Below are your results from the math problem-solving session:")
        self.sm.append_left("$")
        
        self.sm.append_left(self.generate_summary_table(rs, stats))
        self.sm.append_left(self.generate_summary_footer(stats))
        
        # Add closing message
        self.sm.append_left("$")
        self.sm.append_left("$  Your participation is greatly appreciated. You may now close this window.")
        self.sm.append_left("$  Please return to Prolific to complete your submission.")
    
    def calculate_stats(self, results):
        return {
            "time_sum": sum(x["time_spent"] for x in results),
            "correct": sum(1 for x in results if x["is_correct"]),
            "total": len(results),
            "avg_certainty": sum(x["certainty"] for x in results)/len(results) if results else 0
        }
    
    def generate_summary_table(self, results, stats):
        lines = []
        lines.append("""$  ┌────┬───────────────────────────┬───────────────────────────┬─────────┬───────────┐
   | #  | Chosen                    | Correct                   | Certain | Time (s)  |
   ├────┼───────────────────────────┼───────────────────────────┼─────────┼───────────┤""")
        
        for i, itm in enumerate(results, start=1):
            lines.append(f"""   | {i:<2} | {itm['chosen_option']:<25} | {itm['correct_option']:<25} | {itm['certainty']:<7} | {itm['time_spent']:9.2f} |""")
        
        return "\n".join(lines)
    
    def generate_summary_footer(self, stats):
        return f"""   └────┴───────────────────────────┴───────────────────────────┴─────────┴───────────┘
    Total: {stats['total']}   Correct: {stats['correct']}   Wrong: {stats['total'] - stats['correct']}   Time: {stats['time_sum']:.2f}s
    Average certainty: {stats['avg_certainty']:.2f}    Group: {"intervention" if self.sm.session["intervention_group"] else "control"}"""

class InputHandler:
    def __init__(self, session_manager):
        self.sm = session_manager
        self.qm = QuestionManager(session_manager)
    
    def handle(self, user_input):
        handlers = {
            "prolific": self.handle_prolific,
            "questions": self.handle_questions
        }
        return handlers.get(self.sm.session["phase"], lambda x: None)(user_input)
    
    def handle_prolific(self, u):
        if len(u) >= 24:
            self.sm.session.update({
                "prolific_id": u,
                "phase": "questions",
                "idx": 0,
                "results": [],
                "certainty_pending": False
            })
            self.sm.clear_console()
            self.qm.show_question()
        else:
            self.sm.append_left("$  Invalid Prolific ID. Please try again:")
    
    def handle_questions(self, u):
        if self.sm.session["certainty_pending"]:
            if u == "timeout":
                self.record_certainty(1)
                self.advance_question()
                return
            return self.handle_certainty(u)
        
        if not u:
            return
            
        i = self.sm.session["idx"]
        tasks = load_tasks()
        if i >= len(tasks):
            self.end_questions()
            return
            
        self.process_answer(u)
    
    def handle_certainty(self, u):
        if u not in ["1", "2", "3", "4"]:
            self.sm.append_left("$  Invalid input. Please enter a number between 1-4:")
            return
            
        self.record_certainty(int(u))
        self.advance_question()
    
    def process_answer(self, u):
        self.sm.record_keeper.add_user_input(u)
        t = load_tasks()[self.sm.session["idx"]]
        spent = time.time() - self.sm.session["start_time"] if self.sm.session["start_time"] else 0
        
        if u == "timeout" or spent >= QUESTION_TIME_SECONDS:
            return self.handle_timeout(t, spent)
            
        if not self.is_valid_answer(u):
            return
            
        self.record_answer(u, t, spent)
        self.ask_certainty()
    
    def is_valid_answer(self, u):
        allowed_letters = [chr(ord('a') + x) for x in range(NUM_ANSWERS)]
        if u not in allowed_letters:
            self.sm.append_left("$  Invalid letter.")
            return False
        return True

    def handle_timeout(self, task, spent):
        self.sm.append_left("$  TIME'S UP! Moving to next question...")
        self.sm.session["current_result"] = {
            "question": task["question"],
            "chosen_option": "TIMEOUT",
            "correct_option": task["correct_solution"],
            "is_correct": False,
            "time_spent": QUESTION_TIME_SECONDS
        }
        
        self.sm.session["certainty_pending"] = True
        self.ask_certainty()

    def record_answer(self, u, task, spent):
        answer_idx = ord(u) - ord('a')
        chosen = self.sm.session['current_options'][answer_idx]
        correct = (chosen == task["correct_solution"])

        self.sm.session["current_result"] = {
            "question": task["question"],
            "chosen_option": chosen,
            "correct_option": task["correct_solution"],
            "is_correct": correct,
            "time_spent": spent
        }
        
    def ask_certainty(self):
        self.sm.session["certainty_pending"] = True
        self.sm.append_left("\n$  How certain are you about your decision?")
        self.sm.append_left("\n$  Type a digit from 1 to 4 and press ENTER:")
        self.sm.append_left("   1) uncertain")
        self.sm.append_left("   2) rather uncertain")
        self.sm.append_left("   3) rather certain")
        self.sm.append_left("   4) certain")

    def record_certainty(self, certainty):
        self.sm.session["current_result"]["certainty"] = certainty
        self.sm.session["results"].append(self.sm.session["current_result"])
        result = self.sm.session["current_result"]

        self.sm.record_keeper.complete_task(
            result["chosen_option"],
            certainty,
            result["time_spent"]
        )
        self.sm.session["current_result"] = None
        self.sm.session["certainty_pending"] = False

    def advance_question(self):
        self.sm.session["idx"] += 1
        if self.sm.session["idx"] >= len(load_tasks()):
            self.sm.session["phase"] = "summary"
            self.sm.clear_console()
            self.qm.show_summary()
        else:
            # Only clear the left console
            self.sm.session["lines_left"] = []
            # Only clear the right console when moving to a new question
            self.sm.session["lines_right"] = []
            self.qm.show_question()

    def end_questions(self):
        self.sm.session["phase"] = "summary"
        self.sm.clear_console()
        self.qm.show_summary()

@app.before_request
def before():
    session.permanent = False
    SessionManager(session).init_state()
    # Pre-load tasks to make sure they're cached
    load_tasks()

@app.route("/")
def home():
    sm = SessionManager(session)
    if session["phase"] == "prolific" and not session["lines_left"]:
        sm.clear_console()
        sm.append_left("$  Welcome to the Math Problem Solving Study!")
        sm.append_left("$")
        sm.append_left("$  In this study, you will be asked to solve a series of math problems.")
        sm.append_left("$  Some participants will have access to an AI assistant in the right panel")
        sm.append_left("$  that can provide guidance (but not direct answers).")
        sm.append_left("$")
        # Display time in seconds or minutes appropriately
        if QUESTION_TIME_SECONDS < 60:
            sm.append_left(f"$  You will have {QUESTION_TIME_SECONDS} seconds to solve each problem.")
        else:
            minutes = QUESTION_TIME_SECONDS // 60
            seconds = QUESTION_TIME_SECONDS % 60
            time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
            if seconds:
                time_str += f" and {seconds} second{'s' if seconds > 1 else ''}"
            sm.append_left(f"$  You will have {time_str} to solve each problem.")
        
        sm.append_left("$  After submitting your answer, you'll be asked to rate your confidence.")
        sm.append_left("$")
        sm.append_left("$  Please enter your Prolific ID to begin:")
    return render_template("console.html", 
                         lines_left=session["lines_left"], 
                         lines_right=session["lines_right"],
                         session=session)

@app.route("/status")
def status():
    timer_duration = 0
    if session["phase"] == "questions":
        if not session["certainty_pending"]:
            # Convert seconds to minutes for the timer
            timer_duration = QUESTION_TIME_SECONDS / 60
        else:
            # Convert seconds to minutes for the timer
            timer_duration = CERTAINTY_TIME_SECONDS / 60
    return jsonify({
        "timer_duration": timer_duration,
        "should_reset": True,
        "question_idx": session["idx"],
        "certainty_pending": session.get("certainty_pending", False),
        "phase": session["phase"]
    })

@app.route("/command", methods=["POST"])
def command():
    data = request.json or {}
    raw_input = data.get("input", "")
    user_input = raw_input.get("input", "") if isinstance(raw_input, dict) else raw_input
    user_input = str(user_input).strip()
    
    was_certainty_pending = session.get("certainty_pending", False)
    previous_idx = session.get("idx", 0)
    
    ih = InputHandler(SessionManager(session))
    ih.handle(user_input)
    
    timer_duration = 0
    should_reset = False
    current_idx = session.get("idx", 0)
    
    # Check if we've moved to a new question
    new_question = (previous_idx != current_idx and not session.get("certainty_pending", False))
    
    if session["phase"] == "questions":
        current_certainty_pending = session.get("certainty_pending", False)
        
        if not current_certainty_pending:
            # Convert seconds to minutes for the timer
            timer_duration = QUESTION_TIME_SECONDS / 60
            should_reset = was_certainty_pending or user_input == "timeout"
        else:
            # Convert seconds to minutes for the timer
            timer_duration = CERTAINTY_TIME_SECONDS / 60
            should_reset = (not was_certainty_pending) or user_input == "timeout"
    
    return jsonify({
        "lines_left": session["lines_left"],
        "lines_right": session["lines_right"],
        "timer_duration": timer_duration,
        "should_reset": should_reset,
        "certainty_pending": session.get("certainty_pending", False),
        "new_question": new_question
    })

@app.route("/chat", methods=["POST"])
def chat():
    if session["phase"] != "questions":
        return jsonify({"lines_right": session["lines_right"]})
    
    message = request.json.get("message", "").strip()
    current_question_idx = session["idx"]  # Use the current index directly
    
    # Validate if the message is empty
    if not message:
        return jsonify({"lines_right": session["lines_right"]})
    
    sm = SessionManager(session)
    current_task = load_tasks()[current_question_idx]
    
    # Prepare content list starting with the question text
    content = [
        {"type": "text", "text": f"Current math question: {current_task['question']}\n\nUser question: {message}"}
    ]
    
    # Get image from cache using the key stored in session
    image_key = session.get("current_image_key")
    if image_key:
        base64_image = get_cached_image(image_key)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    
    try:
        # Use the global OpenAI client and model
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a math assistant that helps with the current math problem. For math-related questions, help guide the user toward solving the problem without directly giving the answer."},
                {"role": "system", "content": "If the user asks questions that are not related to the current math problem, respond only with something like: 'I'm sorry, but I can only help with the current math problem' or with a small math joke when it's appropriate, but don't continue with the explaination of the task then. Do not elaborate further on off-topic questions. Don't write anything to the task than."},
                {"role": "system", "content": "If the user is writing something like 'Hello' or 'How are you?', respond with: 'Hello, how I can help you with your math problem?'"},
                {"role": "system", "content": "Use only UTF-8 text characters for your results. When you try to write math equations, use plain text instead of special code, e.g. • for \\times or (25/5) instead of \\frac{25}{5}."},
                {"role": "user", "content": content}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        assistant_response = response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        assistant_response = f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    # Format the assistant's response with HTML
    formatted_response = f"<span class='assistant'>Assistant: {assistant_response}</span>"
    
    # Add to session
    sm.append_right(formatted_response)
    
    # Add to history record
    sm.record_keeper.add_chat_interaction(message, assistant_response)
    
    # Return the updated right console lines and the latest response
    return jsonify({
        "lines_right": session["lines_right"],
        "latest_response": formatted_response,
        "question_idx": current_question_idx
    })

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))