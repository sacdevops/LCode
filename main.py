from flask import Flask, render_template, request, jsonify, session
import csv
import os
import time
import secrets
import random
import json
import shutil
from google import genai
from google.genai import types
from dotenv import load_dotenv
from webdav3.client import Client
import config

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

LLM_MODEL = os.getenv('LLM_ENGINE', 'gemini-2.0-flash')
genai_client = genai.Client(api_key=api_key)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["SESSION_PERMANENT"] = False

task_cache = {"test": None, "main": None}
current_chat_session = None

webdav_options = {
    'webdav_hostname': os.getenv('SCIEBO_URL'),
    'webdav_login': os.getenv('SCIEBO_LOGIN'),
    'webdav_password': os.getenv('SCIEBO_PASSWORD'),
    'connect_timeout': 10,
    'read_timeout': 30
}

webdav_client = Client(webdav_options)

def load_tasks():
    if task_cache["test"] is None or task_cache["main"] is None:
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

            task_cache["test"] = test_tasks[:config.TEST_TASKS_COUNT]
            task_cache["main"] = main_tasks[:config.MAIN_TASKS_COUNT]

        except Exception as e:
            print(f"Error loading tasks: {e}")
            task_cache["test"] = []
            task_cache["main"] = []
            
    return task_cache

def prepare_options(task):
    options = task['options'].copy()
    correct = task['correct_solution']
    if correct in options:
        options.remove(correct)
    random.shuffle(options)
    selected = options[:config.NUM_ANSWERS-1]
    final = selected + [correct]
    random.shuffle(final)
    return final

def init_session():
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
        "treatment_group": config.TREATMENT_GROUP,
        "current_phase": "test",
        "waiting_start_time": None,
        "record_data": {"records": [], "current_task": None},
        "session_id": secrets.token_hex(16),
        "current_options": [],
        "current_task_key": None,
        "is_first_message": False
    }
    
    for key, value in defaults.items():
        if key not in session:
            session[key] = value

def clear_console():
    session["lines_left"] = []
    session["lines_right"] = []

def append_left(txt):
    if "lines_left" not in session:
        session["lines_left"] = []
    session["lines_left"].append(txt)

def append_right(txt):
    if "lines_right" not in session:
        session["lines_right"] = []
    session["lines_right"].append(txt)

def get_current_task():
    tasks = load_tasks()
    phase = session["current_phase"]
    idx = session["test_idx"] if phase == "test" else session["main_idx"]
    
    if idx < len(tasks[phase]):
        return tasks[phase][idx]
    return None

def is_phase_complete():
    tasks = load_tasks()
    phase = session["current_phase"]
    idx = session["test_idx"] if phase == "test" else session["main_idx"]
    return idx >= len(tasks[phase])

def show_question():
    global current_chat_session
    
    clear_console()
    session["start_time"] = time.time()
    
    session["lines_right"] = []
    current_chat_session = None
    session["is_first_message"] = False
    
    if is_phase_complete():
        handle_phase_completion()
        return
    
    task = get_current_task()
    if not task:
        return
        
    phase = session["current_phase"]
    idx = session["test_idx"] if phase == "test" else session["main_idx"]
    total = config.TEST_TASKS_COUNT if phase == "test" else config.MAIN_TASKS_COUNT
    prefix = "PRACTICE QUESTION" if phase == "test" else "QUESTION"
    
    session["current_task_key"] = f"{phase}_{idx}"
    
    append_left(f"$  {prefix} {idx+1}/{total}\n")
    
    if task.get("image_path"):
        img_path = os.path.join("static", "img", task["image_path"] + ".jpg")
        if os.path.exists(img_path):
            append_left(f'<img src="static/img/{task["image_path"]}.jpg" class="task-img">')
    
    append_left(f"   {task['question']}")
    
    options = prepare_options(task)
    session['current_options'] = options
    
    start_task_record(idx, task, options, phase)
    
    for x, op in enumerate(options):
        append_left(f"   {chr(ord('a') + x)}) {op}")
    
    append_left("$  Type a letter and press ENTER.")
    
    fixed_greeting = "<span class='assistant'>Assistant: Hello! I'm your mathematical assistant. I'm ready to help you with this math problem.</span>"
    append_right(fixed_greeting)

def handle_phase_completion():
    phase = session["current_phase"]
    
    if phase == "test":
        session["phase"] = "waiting"
        session["waiting_start_time"] = time.time()
        clear_console()
        
        append_left("$  Practice Phase Completed!")
        append_left("$")
        append_left("$  You have completed the practice questions.")
        append_left("$  The main study will begin shortly.")
        append_left("$")
        append_left(f"$  Please wait {config.WAITING_TIME_SECONDS} seconds...")
    else:
        session["phase"] = "summary"
        clear_console()
        show_summary()

def show_waiting_phase():
    elapsed = time.time() - session["waiting_start_time"]
    remaining = max(0, config.WAITING_TIME_SECONDS - elapsed)
    
    if remaining <= 0:
        session["phase"] = "questions"
        session["current_phase"] = "main"
        clear_console()
        show_question()

def show_summary():
    main_results = [r for r in session["results"] if r.get("task_type", "main") == "main"]
    stats = calculate_stats(main_results)
    save_results(stats)
    
    append_left("$  Thank you for participating in our study!")
    append_left("$")
    append_left("$  Below are your results from the math problem-solving session:")
    append_left("$")
    
    append_left(generate_summary_table(main_results))
    append_left(generate_summary_footer(stats))
    
    append_left("$")
    append_left("$  Your participation is greatly appreciated. You may now close this tab.")
    append_left("$  Please return to LimeSurvey to complete the study.")

def calculate_stats(results):
    if not results:
        return {"time_sum": 0, "correct": 0, "total": 0, "avg_certainty": 0, "avg_time": 0}
        
    return {
        "time_sum": sum(x["time_spent"] for x in results),
        "correct": sum(1 for x in results if x["is_correct"]),
        "total": len(results),
        "avg_certainty": sum(x["certainty"] for x in results)/len(results),
        "avg_time": sum(x["time_spent"] for x in results)/len(results),
    }

def generate_summary_table(results):
    lines = []
    lines.append("$  ┌────┬───────────────────────────┬───────────────────────────┬───────────┬───────────┐")
    lines.append("   | #  | Chosen                    | Correct                   | Certainty | Time (s)  |")
    lines.append("   ├────┼───────────────────────────┼───────────────────────────┼───────────┼───────────┤")
    
    for i, itm in enumerate(results, start=1):
        lines.append(f"   | {i:<2} | {itm['chosen_option']:<25} | {itm['correct_option']:<25} | {itm['certainty']:<9} | {itm['time_spent']:9.2f} |")
    
    return "\n".join(lines)

def generate_summary_footer(stats):
    return f"""   └────┴───────────────────────────┴───────────────────────────┴───────────┴───────────┘
    Total: {stats['total']}
    Correct: {stats['correct']}
    Wrong: {stats['total'] - stats['correct']}
    Average time: {stats['avg_time']:.2f}s
    Average certainty: {stats['avg_certainty']:.2f}
    Group: {"treatment" if session["treatment_group"] else "control"}"""

def start_task_record(task_idx, task, options, task_type):
    session["record_data"]["current_task"] = {
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

def add_chat_interaction(user_msg, assistant_msg):
    if session["record_data"]["current_task"]:
        session["record_data"]["current_task"]["chat_history"].append({
            "timestamp": time.time(),
            "user": user_msg,
            "assistant": assistant_msg
        })

def complete_task(answer, certainty, time_spent):
    if session["record_data"]["current_task"]:
        current = session["record_data"]["current_task"]
        current.update({
            "final_answer": answer,
            "certainty": certainty,
            "time_spent": time_spent
        })
        session["record_data"]["records"].append(current)
        session["record_data"]["current_task"] = None

def save_results(stats):
    final_data = {
        "prolific_id": session["prolific_id"],
        "group": "treatment" if session["treatment_group"] else "control",
        "test_tasks": [r for r in session["record_data"]["records"] if r["task_type"] == "test"],
        "main_tasks": [r for r in session["record_data"]["records"] if r["task_type"] == "main"],
        "statistics": stats
    }
    
    try:
        filename = f'results_{session["prolific_id"]}.json'
        filepath = os.path.join("results", filename)
        
        os.makedirs("results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        if webdav_client:
            upload_to_sciebo(filepath, filename)
        
    except Exception as e:
        print(f"Error saving results: {e}")

def upload_to_sciebo(filepath, filename):
    if not webdav_client:
        return
        
    remote_path = os.path.join(os.getenv('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
    
    for attempt in range(3):
        try:
            webdav_client.upload_file(remote_path=remote_path, local_path=filepath)
            os.remove(filepath)
            return
        except Exception:
            if attempt < 2:
                time.sleep(2)
    
    backup_path = filepath.replace(".json", f"_backup_{int(time.time())}.json")
    try:
        shutil.copy(filepath, backup_path)
    except Exception:
        pass

def handle_input(user_input):
    if session["phase"] == "prolific":
        if user_input.strip():
            session.update({
                "prolific_id": user_input.strip(),
                "phase": "questions",
                "current_phase": "test",
                "test_idx": 0,
                "main_idx": 0,
                "results": [],
                "certainty_pending": False
            })
            show_question()
        else:
            append_left("$  Invalid Prolific ID. Please try again:")
    
    elif session["phase"] == "waiting":
        show_waiting_phase()
    
    elif session["phase"] == "questions":
        if session["certainty_pending"]:
            handle_certainty(user_input)
        else:
            handle_answer(user_input)

def handle_answer(user_input):
    if user_input == "timeout":
        handle_timeout()
        return
    
    valid_letters = [chr(ord('a') + x) for x in range(config.NUM_ANSWERS)]
    if not user_input or user_input.lower() not in valid_letters:
        if user_input:
            append_left("$  Invalid letter.")
        return
    
    task = get_current_task()
    if not task:
        return
    
    spent = time.time() - session["start_time"] if session["start_time"] else 0
    
    if spent >= config.QUESTION_TIME_SECONDS:
        handle_timeout()
        return
    
    answer_idx = ord(user_input.lower()) - ord('a')
    if answer_idx >= len(session.get('current_options', [])):
        append_left("$  Invalid option.")
        return
        
    chosen = session['current_options'][answer_idx]
    correct = (chosen == task["correct_solution"])

    session["current_result"] = {
        "question": task["question"],
        "chosen_option": chosen,
        "correct_option": task["correct_solution"],
        "is_correct": correct,
        "time_spent": spent,
        "task_type": session["current_phase"]
    }
    
    ask_certainty()

def handle_timeout():
    task = get_current_task()
    if not task:
        return
    
    if session.get("certainty_pending", False):
        session["current_result"]["certainty"] = 0
        session["results"].append(session["current_result"])
        
        complete_task(
            session["current_result"]["chosen_option"],
            0,
            session["current_result"]["time_spent"]
        )
        
        session["current_result"] = None
        session["certainty_pending"] = False
        
        append_left("$  TIME'S UP! Moving to next question...")
        
        session["lines_right"] = []
        
        advance_question()
        return
        
    session["current_result"] = {
        "question": task["question"],
        "chosen_option": "TIMEOUT",
        "correct_option": task["correct_solution"],
        "is_correct": False,
        "time_spent": config.QUESTION_TIME_SECONDS,
        "task_type": session["current_phase"]
    }
    
    append_left("$  TIME'S UP! Moving to next question...")
    ask_certainty()

def ask_certainty():
    session["certainty_pending"] = True
    
    chosen_option = session['current_result']['chosen_option']
    current_options = session.get('current_options', [])
    
    if chosen_option != "TIMEOUT" and chosen_option in current_options:
        answer_idx = current_options.index(chosen_option)
        letter = chr(ord('a') + answer_idx)
        append_left(f"\n$  You selected {letter}) {chosen_option}")
    else:
        append_left(f"\n$  Time expired - no answer selected")
    
    append_left("\n$  How certain are you about your decision?")
    append_left("\n$  Type a digit from 1 to 4 and press ENTER:")
    append_left("   1) uncertain")
    append_left("   2) rather uncertain")
    append_left("   3) rather certain")
    append_left("   4) certain")

def handle_certainty(user_input):
    if user_input == "timeout":
        certainty = 0
    elif user_input in ["1", "2", "3", "4"]:
        certainty = int(user_input)
    else:
        append_left("$  Invalid input. Please enter a number between 1-4:")
        return
    
    session["current_result"]["certainty"] = certainty
    session["results"].append(session["current_result"])
    
    complete_task(
        session["current_result"]["chosen_option"],
        certainty,
        session["current_result"]["time_spent"]
    )
    
    session["current_result"] = None
    session["certainty_pending"] = False
    advance_question()

def advance_question():
    global current_chat_session
    
    phase = session["current_phase"]
    
    if phase == "test":
        session["test_idx"] += 1
    else:
        session["main_idx"] += 1
    
    session["lines_right"] = []
    session["current_task_key"] = None
    current_chat_session = None
    session["is_first_message"] = False
    
    if is_phase_complete():
        handle_phase_completion()
    else:
        show_question()

@app.before_request
def before_request():
    session.permanent = False
    init_session()
    load_tasks()

@app.route("/")
def home():
    if session["phase"] == "prolific" and not session["lines_left"]:
        time_str = f"{config.QUESTION_TIME_SECONDS} seconds"
        if config.QUESTION_TIME_SECONDS >= 60:
            minutes = config.QUESTION_TIME_SECONDS // 60
            seconds = config.QUESTION_TIME_SECONDS % 60
            time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
            if seconds:
                time_str += f" and {seconds} second{'s' if seconds > 1 else ''}"

        clear_console()
        append_left("$  Welcome to the Math Problem Solving Study!")
        append_left("$")
        append_left(f"$  In the following, you will be asked to solve overall seven math problems that will appear here on the left panel. You will have {time_str} for each task. The answer mode is single-choice, for each problem, four answer options will be shown. After each answer, you will be asked about your confidence.")
        append_left("$")
        append_left("$  You will have access to the <b>mathematical assistant</b>, a chatbot specialized to support your problem-solving. The mathematical assistant will be shown on the right panel. Please use the mathematical assistant to solve all math problems. You may additionally use pen and paper.")
        append_left("$")
        append_left(f"$  The first {config.TEST_TASKS_COUNT} math problems will be part of a test phase, where you can get used to the tool. After that, you will solve {config.MAIN_TASKS_COUNT} math problems as part of the study, that is we will measure and analyse your performance.")
        append_left("$")
        append_left("$  After submitting your answer, you'll be asked to rate your confidence.")
        append_left("$")
        append_left("$  Please enter your Prolific ID to begin:")
        
    return render_template("console.html", 
                         lines_left=session.get("lines_left", []), 
                         lines_right=session.get("lines_right", []),
                         session=session)

@app.route("/status")
def status():
    timer_duration = 0
    if session["phase"] == "questions":
        if not session["certainty_pending"]:
            timer_duration = config.QUESTION_TIME_SECONDS / 60
        else:
            timer_duration = config.CERTAINTY_TIME_SECONDS / 60
    elif session["phase"] == "waiting":
        elapsed = time.time() - session["waiting_start_time"] if session["waiting_start_time"] else 0
        remaining = max(0, config.WAITING_TIME_SECONDS - elapsed)
        timer_duration = remaining / 60
    
    return jsonify({
        "timer_duration": timer_duration,
        "should_reset": True,
        "question_idx": session["test_idx"] if session["current_phase"] == "test" else session["main_idx"],
        "certainty_pending": session.get("certainty_pending", False),
        "phase": session["phase"],
        "waiting_phase": session["phase"] == "waiting",
        "lines_right": session.get("lines_right", [])
    })

@app.route("/command", methods=["POST"])
def command():
    try:
        data = request.json or {}
        raw_input = data.get("input", "")
        user_input = raw_input.get("input", "") if isinstance(raw_input, dict) else raw_input
        user_input = str(user_input).strip().lower()
        
        was_certainty_pending = session.get("certainty_pending", False)
        previous_test_idx = session.get("test_idx", 0)
        previous_main_idx = session.get("main_idx", 0)
        previous_phase = session.get("current_phase", "test")
        
        handle_input(user_input)
        
        current_phase = session.get("current_phase", "test")
        current_test_idx = session.get("test_idx", 0)
        current_main_idx = session.get("main_idx", 0)
        
        new_question = ((previous_test_idx != current_test_idx and current_phase == "test") or 
                       (previous_main_idx != current_main_idx and current_phase == "main") or
                       previous_phase != current_phase) and not session.get("certainty_pending", False)
        
        timer_duration = 0
        should_reset = False
        
        if session["phase"] == "questions":
            current_certainty_pending = session.get("certainty_pending", False)
            
            if not current_certainty_pending:
                timer_duration = config.QUESTION_TIME_SECONDS / 60
                should_reset = was_certainty_pending or user_input == "timeout" or new_question
            else:
                timer_duration = config.CERTAINTY_TIME_SECONDS / 60
                should_reset = (not was_certainty_pending) or user_input == "timeout"
        elif session["phase"] == "waiting":
            elapsed = time.time() - session["waiting_start_time"] if session["waiting_start_time"] else 0
            remaining = max(0, config.WAITING_TIME_SECONDS - elapsed)
            timer_duration = remaining / 60
            should_reset = True
        
        return jsonify({
            "lines_left": session.get("lines_left", []),
            "lines_right": session.get("lines_right", []),
            "timer_duration": timer_duration,
            "should_reset": should_reset,
            "certainty_pending": session.get("certainty_pending", False),
            "new_question": new_question,
            "waiting_phase": session["phase"] == "waiting",
            "phase": session["phase"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    global current_chat_session
    
    try:
        if (session["phase"] != "questions" or 
            session["phase"] == "waiting"):
            return jsonify({
                "lines_right": session.get("lines_right", []), 
                "error": "Chat not available"
            })
        
        message = request.json.get("message", "").strip()
        if not message:
            return jsonify({"lines_right": session.get("lines_right", []), "error": "Empty message"})
        
        user_message_formatted = f"<span class='user'>You: {message}</span>"
        append_right(user_message_formatted)
        
        task_key = session.get("current_task_key")
        if not task_key:
            return jsonify({"error": "No active task", "lines_right": session.get("lines_right", [])})
        
        is_first_message = not session.get("is_first_message", False)
        
        if is_first_message or current_chat_session is None:
            current_chat_session = genai_client.chats.create(
                model=LLM_MODEL,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0,
                )
            )
            
            contents = []
            treatment_group = session["treatment_group"]
            prompt = config.TREATMENT_GROUP_PROMPT if treatment_group else config.CONTROL_GROUP_PROMPT
            contents.append(prompt)
            
            task = get_current_task()
            if task and task.get('image_path'):
                img_path = os.path.join("static", "img", task['image_path'] + ".jpg")
                if os.path.exists(img_path):
                    try:
                        uploaded_file = genai_client.files.upload(
                            file=img_path,
                            config=types.UploadFileConfig(mime_type="image/jpeg")
                        )
                        contents.append(uploaded_file)
                    except Exception as e:
                        print(f"Error uploading image {img_path}: {e}")
                else:
                    print(f"Image file not found: {img_path}")
            
            contents.append(f"Current math question: {task['question']}")
            contents.append(message)
            
            response = current_chat_session.send_message(contents)
            
            session["is_first_message"] = True
        else:
            response = current_chat_session.send_message(message)
        
        assistant_response = response.text
        assistant_message_formatted = f"<span class='assistant'>Assistant: {assistant_response}</span>"
        append_right(assistant_message_formatted)
        
        add_chat_interaction(message, assistant_response)
        
        return jsonify({
            "lines_right": session.get("lines_right", []),
            "latest_response": assistant_message_formatted
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
