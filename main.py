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
from config import NUM_ANSWERS, CERTAINTY_TIME_SECONDS, QUESTION_TIME_SECONDS, TEST_TASKS_COUNT, MAIN_TASKS_COUNT, WAITING_TIME_SECONDS

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
    
LLM_MODEL = os.getenv('LLM_ENGINE', 'gpt-4.1-mini')

openai_client = openai.OpenAI(api_key=api_key)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["SESSION_PERMANENT"] = False

cache = {
    "images": {},
    "tasks": {
        "test": None,
        "main": None
    }
}

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
        
    def start_task_record(self, task_idx, task, options, task_type):
        self.session["record_data"]["current_task"] = {
            "task_index": task_idx,
            "question": task["question"],
            "options": options,
            "solution": task["correct_solution"],
            "user_inputs": [],
            "chat_history": [],
            "time_spent": 0,
            "final_answer": None,
            "certainty": None,
            "task_type": task_type
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
            "test_tasks": [r for r in self.session["record_data"]["records"] if r["task_type"] == "test"],
            "main_tasks": [r for r in self.session["record_data"]["records"] if r["task_type"] == "main"],
            "statistics": stats
        }
        
        try:
            filename = f'results_{self.session["prolific_id"]}.json'
            filepath = os.path.join("results", filename)
            
            os.makedirs("results", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, ensure_ascii=False, fp=f)

            remote_path = os.path.join(os.getenv('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
            
            # Implement retry mechanism for upload
            max_retries = 5
            retry_count = 0
            upload_successful = False
            
            while not upload_successful and retry_count < max_retries:
                try:
                    print(f"Uploading results file, attempt {retry_count + 1}/{max_retries}")
                    webdav_client.upload_file(
                        remote_path=remote_path,
                        local_path=filepath
                    )
                    upload_successful = True
                    print("Upload successful!")
                except Exception as upload_error:
                    retry_count += 1
                    error_message = str(upload_error)
                    print(f"Upload attempt {retry_count} failed: {error_message}")
                    
                    if retry_count < max_retries:
                        # Exponential backoff: wait longer with each retry
                        wait_time = 1 * (2 ** (retry_count - 1))  # 1, 2, 4, 8, 16 seconds
                        print(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"Maximum retry attempts ({max_retries}) reached. Upload failed.")
                        # Keep the local file as backup if upload ultimately fails
                        backup_path = filepath.replace(".json", f"_backup_{int(time.time())}.json")
                        try:
                            import shutil
                            shutil.copy(filepath, backup_path)
                            print(f"Backup file saved to {backup_path}")
                        except Exception as backup_error:
                            print(f"Failed to create backup file: {str(backup_error)}")

            # Only remove the file if upload was successful
            if upload_successful:
                os.remove(filepath)
            
        except Exception as e:
            print(f"Error in save_and_send: {str(e)}")

def load_tasks():
    if cache["tasks"]["test"] is None or cache["tasks"]["main"] is None:
        test_tasks = []
        main_tasks = []
        
        csv_file = os.path.join("data", "tasks.csv")
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    task = {
                        "question": row["question"].strip(),
                        "options": [x.strip() for x in row["options"].split(";")],
                        "correct_solution": row["correct_solution"].strip(),
                        "image_path": row.get("image_path", "").strip(),
                        "task_type": row.get("task_type", "main").strip()
                    }
                    if task["correct_solution"] not in task["options"]:
                        task["options"].append(task["correct_solution"])
                    
                    if task["task_type"] == "test":
                        test_tasks.append(task)
                    else:
                        main_tasks.append(task)
                        
            random.shuffle(test_tasks)
            random.shuffle(main_tasks)
            
            test_tasks = test_tasks[:TEST_TASKS_COUNT]
            main_tasks = main_tasks[:MAIN_TASKS_COUNT]
            
            cache["tasks"]["test"] = test_tasks
            cache["tasks"]["main"] = main_tasks
            
        except Exception as e:
            print(f"Error loading tasks: {e}")
            cache["tasks"]["test"] = []
            cache["tasks"]["main"] = []
            
    return {
        "test": cache["tasks"]["test"],
        "main": cache["tasks"]["main"]
    }

def prepare_options(task):
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
    if not image_path:
        return None
        
    cache_key = "current_task_image"
    
    img_path = os.path.join("static", "img", image_path + ".jpg")
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as img_file:
                image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                cache["images"] = {cache_key: base64_image}
                return cache_key
        except Exception as e:
            print(f"Error caching image: {str(e)}")
    
    return None

def get_cached_image(cache_key):
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
            "test_idx": 0,
            "main_idx": 0,
            "results": [],
            "start_time": None,
            "prolific_id": None,
            "certainty_pending": False,
            "current_result": None,
            "intervention_group": random.choice([True, False]),
            "current_image_key": None,
            "current_phase": "test",
            "waiting_start_time": None
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
        phase = self.sm.session["current_phase"]
        if phase == "test":
            i = self.sm.session["test_idx"]
            tlist = load_tasks()["test"]
            total_count = TEST_TASKS_COUNT
            prefix = "PRACTICE QUESTION"
        else:
            i = self.sm.session["main_idx"]
            tlist = load_tasks()["main"]
            total_count = MAIN_TASKS_COUNT
            prefix = "QUESTION"
        
        if i >= len(tlist):
            if phase == "test":
                self.sm.session["phase"] = "waiting"
                self.sm.session["waiting_start_time"] = time.time()
                self.sm.clear_console()
                
                self.sm.append_left("$  Practice Phase Completed!")
                self.sm.append_left("$")
                self.sm.append_left("$  You have completed the practice questions.")
                self.sm.append_left("$  The main study will begin shortly.")
                self.sm.append_left("$")
                self.sm.append_left(f"$  Please wait {WAITING_TIME_SECONDS} seconds...")
                
                return
            else:
                self.sm.session["phase"] = "summary"
                self.sm.clear_console()
                return self.show_summary()
        
        self.sm.session["start_time"] = time.time()
        t = tlist[i]
        
        self.sm.append_left(f"$  {prefix} {i+1}/{total_count}\n")
        
        self.sm.session["current_image_key"] = None
        if t["image_path"]:
            img_path = os.path.join("static", "img", t["image_path"] + ".jpg")
            if os.path.exists(img_path):
                self.sm.append_left(f"""<img src="static/img/{t['image_path']}.jpg" class="task-img">""")
                
                self.sm.session["current_image_key"] = cache_image(i, t["image_path"])
        
        self.sm.append_left(f"   {t['question']}")
        
        options = prepare_options(t)
        self.sm.session['current_options'] = options
        
        if "record_data" not in self.sm.session:
            self.sm.session["record_data"] = {
                "records": [],
                "current_task": None
            }
        self.sm.record_keeper.start_task_record(i, t, options, phase)
        
        for x, op in enumerate(options):
            self.sm.append_left(f"   {chr(ord('a') + x)}) {op}")
        
        self.sm.append_left("$  Type a letter and press ENTER.")
    
    def show_waiting_phase(self):
        elapsed = time.time() - self.sm.session["waiting_start_time"]
        remaining = max(0, WAITING_TIME_SECONDS - elapsed)
        
        if remaining <= 0:
            self.sm.session["phase"] = "questions"
            self.sm.session["current_phase"] = "main"
            self.sm.clear_console()
            self.show_question()
    
    def show_summary(self):
        main_results = [r for r in self.sm.session["results"] if r.get("task_type", "main") == "main"]
        stats = self.calculate_stats(main_results)
        self.sm.record_keeper.save_and_send(stats)
        
        self.sm.append_left("$  Thank you for participating in our study!")
        self.sm.append_left("$")
        self.sm.append_left("$  Below are your results from the math problem-solving session:")
        self.sm.append_left("$")
        
        self.sm.append_left(self.generate_summary_table(main_results, stats))
        self.sm.append_left(self.generate_summary_footer(stats))
        
        self.sm.append_left("$")
        self.sm.append_left("$  Your participation is greatly appreciated. You may now close this tab.")
        self.sm.append_left("$  Please return to LimeSurvey to complete the study.")
    
    def calculate_stats(self, results):
        return {
            "time_sum": sum(x["time_spent"] for x in results),
            "correct": sum(1 for x in results if x["is_correct"]),
            "total": len(results),
            "avg_certainty": sum(x["certainty"] for x in results)/len(results) if results else 0,
            "avg_time": sum(x["time_spent"] for x in results)/len(results) if results else 0,
        }
    
    def generate_summary_table(self, results, stats):
        lines = []
        lines.append("""$  ┌────┬───────────────────────────┬───────────────────────────┬───────────┬───────────┐
   | #  | Chosen                    | Correct                   | Certainty | Time (s)  |
   ├────┼───────────────────────────┼───────────────────────────┼───────────┼───────────┤""")
        
        for i, itm in enumerate(results, start=1):
            lines.append(f"""   | {i:<2} | {itm['chosen_option']:<25} | {itm['correct_option']:<25} | {itm['certainty']:<9} | {itm['time_spent']:9.2f} |""")
        
        return "\n".join(lines)
    
    def generate_summary_footer(self, stats):
        return f"""   └────┴───────────────────────────┴───────────────────────────┴───────────┴───────────┘
    Total: {stats['total']}
    Correct: {stats['correct']}
    Wrong: {stats['total'] - stats['correct']}
    Average time: {stats['avg_time']:.2f}s
    Average certainty: {stats['avg_certainty']:.2f}
    Group: {"intervention" if self.sm.session["intervention_group"] else "control"}"""

class InputHandler:
    def __init__(self, session_manager):
        self.sm = session_manager
        self.qm = QuestionManager(session_manager)
    
    def handle(self, user_input):
        handlers = {
            "prolific": self.handle_prolific,
            "questions": self.handle_questions,
            "waiting": self.handle_waiting
        }
        return handlers.get(self.sm.session["phase"], lambda x: None)(user_input)
    
    def handle_prolific(self, u):
        if len(u) > 0:
            self.sm.session.update({
                "prolific_id": u,
                "phase": "questions",
                "current_phase": "test",
                "test_idx": 0,
                "main_idx": 0,
                "results": [],
                "certainty_pending": False
            })
            self.sm.clear_console()
            self.qm.show_question()
        else:
            self.sm.append_left("$  Invalid Prolific ID. Please try again:")
    
    def handle_waiting(self, u):
        self.qm.show_waiting_phase()
        return
    
    def handle_questions(self, u):
        if self.sm.session["certainty_pending"]:
            if u == "timeout":
                self.record_certainty(1)
                self.advance_question()
                return
            return self.handle_certainty(u)
        
        if not u:
            return
            
        phase = self.sm.session["current_phase"]
        if phase == "test":
            i = self.sm.session["test_idx"]
            tasks = load_tasks()["test"]
        else:
            i = self.sm.session["main_idx"]
            tasks = load_tasks()["main"]
            
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
        
        phase = self.sm.session["current_phase"]
        if phase == "test":
            i = self.sm.session["test_idx"]
            t = load_tasks()["test"][i]
        else:
            i = self.sm.session["main_idx"]
            t = load_tasks()["main"][i]
            
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
            "time_spent": QUESTION_TIME_SECONDS,
            "task_type": self.sm.session["current_phase"]
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
            "time_spent": spent,
            "task_type": self.sm.session["current_phase"]
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
        phase = self.sm.session["current_phase"]
        
        if phase == "test":
            self.sm.session["test_idx"] += 1
            if self.sm.session["test_idx"] >= len(load_tasks()["test"]):
                self.sm.session["phase"] = "waiting"
                self.sm.session["waiting_start_time"] = time.time()
                
                self.sm.session["lines_left"] = []
                self.sm.session["lines_right"] = []
                
                self.sm.append_left("$  Practice Phase Completed!")
                self.sm.append_left("$")
                self.sm.append_left("$  You have completed the practice questions.")
                self.sm.append_left("$  The main study will begin shortly.")
                self.sm.append_left("$")
                self.sm.append_left(f"$  Please wait {WAITING_TIME_SECONDS} seconds...")
            else:
                self.sm.session["lines_left"] = []
                self.qm.show_question()
        else:
            self.sm.session["main_idx"] += 1
            if self.sm.session["main_idx"] >= len(load_tasks()["main"]):
                self.sm.session["phase"] = "summary"
                self.sm.clear_console()
                self.qm.show_summary()
            else:
                self.sm.session["lines_left"] = []
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
    load_tasks()

@app.route("/")
def home():
    sm = SessionManager(session)
    if session["phase"] == "prolific" and not session["lines_left"]:
        time_str = ""
        if QUESTION_TIME_SECONDS < 60:
            time_str = f"{QUESTION_TIME_SECONDS} seconds"
        else:
            minutes = QUESTION_TIME_SECONDS // 60
            seconds = QUESTION_TIME_SECONDS % 60
            time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
            if seconds:
                time_str += f" and {seconds} second{'s' if seconds > 1 else ''}"

        sm.clear_console()
        sm.append_left("$  Welcome to the Math Problem Solving Study!")
        sm.append_left("$")
        sm.append_left(f"$  In the following, you will be asked to solve overall seven math problems that will appear here on the left panel. You will have {time_str} minutes for each task. The answer mode is single-choice, for each problem, four answer options will be shown. After each answer, you will be asked about your confidence in that answer.  ")
        sm.append_left("$")
        sm.append_left("$  You will have access to the <b>mathematical assistant</b>, a chatbot specialized to support your problem-solving. The mathematical assistant will be shown on the right panel. Please use the mathematical assistant to solve all math problems. You may additionally use pen and paper. ")
        sm.append_left("$")
        sm.append_left(f"$  The first {TEST_TASKS_COUNT} math problems will be part of a test phase, where you can get used to the tool. After that, you will solve {MAIN_TASKS_COUNT} math problems as part of the study, that is we will measure and analyse your performance. ")
        sm.append_left("$")
        
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
            timer_duration = QUESTION_TIME_SECONDS / 60
        else:
            timer_duration = CERTAINTY_TIME_SECONDS / 60
    elif session["phase"] == "waiting":
        elapsed = time.time() - session["waiting_start_time"] if session["waiting_start_time"] else 0
        remaining = max(0, WAITING_TIME_SECONDS - elapsed)
        timer_duration = remaining / 60
    
    return jsonify({
        "timer_duration": timer_duration,
        "should_reset": True,
        "question_idx": session["test_idx"] if session["current_phase"] == "test" else session["main_idx"],
        "certainty_pending": session.get("certainty_pending", False),
        "phase": session["phase"],
        "waiting_phase": session["phase"] == "waiting"
    })

@app.route("/command", methods=["POST"])
def command():
    data = request.json or {}
    raw_input = data.get("input", "")
    user_input = raw_input.get("input", "") if isinstance(raw_input, dict) else raw_input
    user_input = str(user_input).strip()
    
    was_certainty_pending = session.get("certainty_pending", False)
    
    previous_test_idx = session.get("test_idx", 0)
    previous_main_idx = session.get("main_idx", 0)
    previous_phase = session.get("current_phase", "test")
    
    ih = InputHandler(SessionManager(session))
    ih.handle(user_input)
    
    timer_duration = 0
    should_reset = False
    
    current_phase = session.get("current_phase", "test")
    current_test_idx = session.get("test_idx", 0)
    current_main_idx = session.get("main_idx", 0)
    
    new_question = ((previous_test_idx != current_test_idx and current_phase == "test") or 
                   (previous_main_idx != current_main_idx and current_phase == "main") or
                   previous_phase != current_phase) and not session.get("certainty_pending", False)
    
    if session["phase"] == "questions":
        current_certainty_pending = session.get("certainty_pending", False)
        
        if not current_certainty_pending:
            timer_duration = QUESTION_TIME_SECONDS / 60
            should_reset = was_certainty_pending or user_input == "timeout"
        else:
            timer_duration = CERTAINTY_TIME_SECONDS / 60
            should_reset = (not was_certainty_pending) or user_input == "timeout"
    elif session["phase"] == "waiting":
        elapsed = time.time() - session["waiting_start_time"] if session["waiting_start_time"] else 0
        remaining = max(0, WAITING_TIME_SECONDS - elapsed)
        timer_duration = remaining / 60
        should_reset = True
    
    return jsonify({
        "lines_left": session["lines_left"],
        "lines_right": session["lines_right"],
        "timer_duration": timer_duration,
        "should_reset": should_reset,
        "certainty_pending": session.get("certainty_pending", False),
        "new_question": new_question,
        "waiting_phase": session["phase"] == "waiting",
        "phase": session["phase"]
    })

@app.route("/chat", methods=["POST"])
def chat():
    if session["phase"] != "questions" or session.get("certainty_pending", False):
        return jsonify({"lines_right": session["lines_right"]})
    
    message = request.json.get("message", "").strip()
    
    if session["current_phase"] == "test":
        current_question_idx = session["test_idx"]
        task_list = load_tasks()["test"]
    else:
        current_question_idx = session["main_idx"]
        task_list = load_tasks()["main"]
    
    if not message or current_question_idx >= len(task_list):
        return jsonify({"lines_right": session["lines_right"]})
    
    sm = SessionManager(session)
    current_task = task_list[current_question_idx]
    
    content = [
        {"type": "text", "text": f"Current math question: {current_task['question']}\n\nUser question: {message}"}
    ]
    
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
    
    formatted_response = f"<span class='assistant'>Assistant: {assistant_response}</span>"
    
    sm.append_right(formatted_response)
    
    sm.record_keeper.add_chat_interaction(message, assistant_response)
    
    return jsonify({
        "lines_right": session["lines_right"],
        "latest_response": formatted_response,
        "question_idx": current_question_idx
    })

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))