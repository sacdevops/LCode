from flask import Flask, render_template, request, jsonify, session
import csv
import os
import time
import secrets
import random
import openai
import json
from webdav3.client import Client

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["SESSION_PERMANENT"] = False
openai.api_key = os.getenv('OPENAI_API_KEY')

class RecordKeeper:
    def __init__(self, session):
        self.session = session
        
    def start_task_record(self, task_idx, task, options):
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
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, indent=2, ensure_ascii=False, fp=f)

            options = {
                'webdav_hostname': os.getenv('SCIEBO_URL'),
                'webdav_login': os.getenv('SCIEBO_LOGIN'),
                'webdav_password': os.getenv('SCIEBO_PASSWORD')
            }
            client = Client(options)
            
            remote_path = os.path.join(os.getenv('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
            
            client.upload_file(
                remote_path=remote_path,
                local_path=filepath
            )
            
            print(f"File successfully uploaded to Sciebo: {remote_path}")
            os.remove(filepath)
        except Exception as e:
            print(f"Error saving/uploading results: {str(e)}")

class TaskLoader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
    
    def load(self):
        result = []
        with open(self.csv_file, "r", encoding="utf-8") as f:
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
        return result

class SessionManager:
    def __init__(self, session):
        self.session = session
        self.record_keeper = RecordKeeper(session)
    
    def init_state(self):
        defaults = {
            "phase": "prolific",
            "lines_left": [],
            "lines_right": [],
            "tasks": TaskLoader(os.path.join("data", "tasks.csv")).load(),
            "idx": 0,
            "results": [],
            "num_answers": 4,
            "start_time": None,
            "prolific_id": None,
            "certainty_pending": False,
            "current_result": None,
            "intervention_group": random.choice([True, False])
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
        self.session["lines_left"].append(txt)
    
    def append_right(self, txt):
        self.session["lines_right"].append(txt)
    
    def reset(self):
        self.session.clear()
        self.init_state()

class QuestionManager:
    def __init__(self, session_manager):
        self.sm = session_manager
    
    def show_question(self):
        i = self.sm.session["idx"]
        tlist = self.sm.session["tasks"]
        if i >= len(tlist):
            self.sm.session["phase"] = "summary"
            self.sm.clear_console()
            return self.show_summary()
        
        self.sm.session["start_time"] = time.time()
        t = tlist[i]
        
        if t["image_path"]:
            img_path = os.path.join("static", "img", t["image_path"] + ".jpg")
            if os.path.exists(img_path):
                self.sm.append_left(f"""<img src="static/img/{t['image_path']}.jpg" style="width:400px;">""")
        
        self.sm.append_left(f"$  QUESTION {i+1}/{len(tlist)}\n")
        self.sm.append_left(f"   {t['question']}")
        
        options = self.prepare_options(t)
        self.sm.session['current_options'] = options
        
        if "record_data" not in self.sm.session:
            self.sm.session["record_data"] = {
                "records": [],
                "current_task": None
            }
        self.sm.record_keeper.start_task_record(i, t, options)
        
        for x, op in enumerate(options):
            self.sm.append_left(f"   {chr(ord('a') + x)}) {op}")
        
        self.sm.append_left("$  Type a letter and press ENTER.")
    
    def prepare_options(self, task):
        options = task['options'].copy()
        correct = task['correct_solution']
        if correct in options:
            options.remove(correct)
        random.shuffle(options)
        selected = options[:self.sm.session["num_answers"]-1]
        final = selected + [correct]
        random.shuffle(final)
        return final
    
    def show_summary(self):
        rs = self.sm.session["results"]
        stats = self.calculate_stats(rs)
        self.sm.record_keeper.save_and_send(stats)
        
        self.sm.append_left(self.generate_summary_table(rs, stats))
        self.sm.append_left(self.generate_summary_footer(stats))
    
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
                "num_answers": 4,
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
        if i >= len(self.sm.session["tasks"]):
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
        t = self.sm.session["tasks"][self.sm.session["idx"]]
        spent = time.time() - self.sm.session["start_time"] if self.sm.session["start_time"] else 0
        
        if u == "timeout" or spent >= 240:
            return self.handle_timeout(t, spent)
            
        if not self.is_valid_answer(u):
            return
            
        self.record_answer(u, t, spent)
        self.ask_certainty()
    
    def is_valid_answer(self, u):
        allowed_letters = [chr(ord('a') + x) for x in range(self.sm.session["num_answers"])]
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
            "time_spent": 240
        }
        if self.sm.session["intervention_group"]:
            self.sm.session["num_answers"] = max(2, self.sm.session["num_answers"] - 1) if random.random() < 0.5 else min(10, self.sm.session["num_answers"] + 1)
        else:
            self.sm.session["num_answers"] = max(2, self.sm.session["num_answers"] - 1)
        self.sm.session["certainty_pending"] = True
        self.ask_certainty()

    def record_answer(self, u, task, spent):
        answer_idx = ord(u) - ord('a')
        chosen = self.sm.session['current_options'][answer_idx]
        correct = (chosen == task["correct_solution"])

        if self.sm.session["intervention_group"]:
            self.sm.session["num_answers"] = max(2, self.sm.session["num_answers"] - 1) if random.random() < 0.5 else min(10, self.sm.session["num_answers"] + 1)
        else:
            if correct:
                self.sm.session["num_answers"] = min(10, self.sm.session["num_answers"] + 1)
            else:
                self.sm.session["num_answers"] = max(2, self.sm.session["num_answers"] - 1)

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
        if self.sm.session["idx"] >= len(self.sm.session["tasks"]):
            self.sm.session["phase"] = "summary"
            self.sm.clear_console()
            self.qm.show_summary()
        else:
            self.sm.clear_console()
            self.qm.show_question()

    def end_questions(self):
        self.sm.session["phase"] = "summary"
        self.sm.clear_console()
        self.qm.show_summary()

@app.before_request
def before():
    session.permanent = False
    SessionManager(session).init_state()

@app.route("/")
def home():
    sm = SessionManager(session)
    if session["phase"] == "prolific" and not session["lines_left"]:
        sm.clear_console()
        sm.append_left("$  Enter your Prolific ID to begin:")
    return render_template("console.html", 
                         lines_left=session["lines_left"], 
                         lines_right=session["lines_right"],
                         session=session)

@app.route("/status")
def status():
    timer_duration = 0
    if session["phase"] == "questions":
        if not session["certainty_pending"]:
            timer_duration = 1
        else:
            timer_duration = 0.5
    return jsonify({
        "timer_duration": timer_duration,
        "should_reset": True
    })

@app.route("/command", methods=["POST"])
def command():
    data = request.json or {}
    raw_input = data.get("input", "")
    user_input = raw_input.get("input", "") if isinstance(raw_input, dict) else raw_input
    user_input = str(user_input).strip()
    
    was_certainty_pending = session.get("certainty_pending", False)
    
    ih = InputHandler(SessionManager(session))
    ih.handle(user_input)
    
    timer_duration = 0
    should_reset = False
    
    if session["phase"] == "questions":
        current_certainty_pending = session.get("certainty_pending", False)
        
        if not current_certainty_pending:
            timer_duration = 1
            should_reset = was_certainty_pending or user_input == "timeout"
        else:
            timer_duration = 0.5
            should_reset = (not was_certainty_pending) or user_input == "timeout"
            
    return jsonify({
        "lines_left": session["lines_left"],
        "lines_right": session["lines_right"],
        "timer_duration": timer_duration,
        "should_reset": should_reset
    })

@app.route("/chat", methods=["POST"])
def chat():
    if session["phase"] != "questions":
        return jsonify({"lines_right": session["lines_right"]})
    
    message = request.json.get("message", "").strip()
    if not message:
        return jsonify({"lines_right": session["lines_right"]})
    
    sm = SessionManager(session)
    sm.append_right(f"<span class='user'>User: {message}</span>")
    
    current_task = session["tasks"][session["idx"]]
    response = openai.chat.completions.create(
        model=os.getenv('LLM_ENGINE', 'gpt-4o-mini'),
        messages=[
            {"role": "system", "content": "Use only UTF-8 text characters for your results. When you try to write math equations, use plain text instead of special code, e.g. • for \times or (25/5) instead of \frac{25}{5}."},
            {"role": "system", "content": "You are a helpful math assistant. Help the user solve the current question without directly giving the answer."},
            {"role": "user", "content": f"Current math question: {current_task['question']}\nUser question: {message}"}
        ],
        max_tokens=1000,
        temperature=0
    )
    
    assistant_response = response.choices[0].message.content
    sm.append_right(f"<span class='assistant'>Assistant: {assistant_response}</span>")
    sm.record_keeper.add_chat_interaction(message, assistant_response)
    
    return jsonify({
        "lines_right": session["lines_right"]
    })

if __name__ == "__main__":
    app.run(debug=False)