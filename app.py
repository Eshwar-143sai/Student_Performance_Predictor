from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import os
import urllib.parse

app = FastAPI(title="Student Performance Predictor API")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Load models at startup
preprocessor_path = 'models/preprocessor.joblib'
best_model_path = 'models/best_model.pkl'

preprocessor = None
model = None

@app.on_event("startup")
def load_models():
    global preprocessor, model
    if os.path.exists(preprocessor_path) and os.path.exists(best_model_path):
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(best_model_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Models not found. Please run train_models.py first.")


class StudentData(BaseModel):
    school: str = "GP"
    sex: str = "F"
    age: int = 18
    address: str = "U"
    famsize: str = "GT3"
    Pstatus: str = "T"
    Medu: int = 2
    Fedu: int = 2
    Mjob: str = "other"
    Fjob: str = "other"
    reason: str = "course"
    guardian: str = "mother"
    traveltime: int = 1
    studytime: int = 2
    failures: int = 0
    schoolsup: str = "no"
    famsup: str = "yes"
    paid: str = "no"
    activities: str = "no"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "yes"
    romantic: str = "no"
    famrel: int = 4
    freetime: int = 3
    goout: int = 3
    Dalc: int = 1
    Walc: int = 1
    health: int = 3
    absences: int = 0
    G1: int = 10
    G2: int = 10


def generate_suggestions(student: StudentData, predicted_grade: float):
    """Generate personalized grade improvement suggestions based on student data."""
    suggestions = []
    
    # Study time suggestions
    if student.studytime <= 1:
        suggestions.append({
            "icon": "📚",
            "title": "Increase Study Time",
            "detail": "You're currently studying less than 2 hours/week. Try to dedicate at least 5-10 hours/week to see significant improvement.",
            "priority": "high"
        })
    elif student.studytime == 2:
        suggestions.append({
            "icon": "📖",
            "title": "Study a Bit More",
            "detail": "You study 2-5 hours/week. Bumping it to 5-10 hours/week can boost your grade noticeably.",
            "priority": "medium"
        })
    
    # Absences
    if student.absences > 10:
        suggestions.append({
            "icon": "🏫",
            "title": "Reduce Absences",
            "detail": f"You have {student.absences} absences. High absenteeism strongly correlates with lower grades. Try not to miss classes.",
            "priority": "high"
        })
    elif student.absences > 5:
        suggestions.append({
            "icon": "📅",
            "title": "Improve Attendance",
            "detail": f"You have {student.absences} absences. Reducing them below 5 can help maintain consistent performance.",
            "priority": "medium"
        })
    
    # Past failures
    if student.failures > 0:
        suggestions.append({
            "icon": "🔄",
            "title": "Address Past Failures",
            "detail": f"You have {student.failures} past class failure(s). Focus on understanding weak areas and seek extra tutoring or support.",
            "priority": "high"
        })
    
    # School support
    if student.schoolsup == "no":
        suggestions.append({
            "icon": "🤝",
            "title": "Seek School Support",
            "detail": "You're not using extra educational support from school. Consider joining tutoring sessions or asking teachers for help.",
            "priority": "medium"
        })
    
    # Internet access
    if student.internet == "no":
        suggestions.append({
            "icon": "🌐",
            "title": "Get Internet Access",
            "detail": "Internet access opens up free learning resources like Khan Academy, YouTube tutorials, and practice problems.",
            "priority": "medium"
        })
    
    # Alcohol consumption
    if student.Dalc >= 3 or student.Walc >= 4:
        suggestions.append({
            "icon": "⚠️",
            "title": "Reduce Alcohol Consumption",
            "detail": "High alcohol consumption is linked to lower academic performance. Try to cut back, especially on weekdays.",
            "priority": "high"
        })
    
    # Going out frequency
    if student.goout >= 4:
        suggestions.append({
            "icon": "🎯",
            "title": "Balance Social Life",
            "detail": "Going out very frequently can eat into study time. Try to balance social activities with academics.",
            "priority": "medium"
        })
    
    # Higher education aspiration
    if student.higher == "no":
        suggestions.append({
            "icon": "🎓",
            "title": "Set Higher Goals",
            "detail": "Students who aspire for higher education tend to perform better. Setting ambitious goals can be a powerful motivator.",
            "priority": "medium"
        })
    
    # Activities
    if student.activities == "no":
        suggestions.append({
            "icon": "🏃",
            "title": "Join Extracurricular Activities",
            "detail": "Extracurricular activities build discipline and time management skills that help academic performance.",
            "priority": "low"
        })

    # Grade-based suggestions
    if predicted_grade < 10:
        suggestions.append({
            "icon": "🆘",
            "title": "Seek Immediate Help",
            "detail": "Your predicted grade is below passing. Consider getting a tutor, forming study groups, and talking to your teachers right away.",
            "priority": "high"
        })
    elif predicted_grade < 14:
        suggestions.append({
            "icon": "📈",
            "title": "Push for Excellence",
            "detail": "You're at an average level. With focused effort on weak areas and consistent practice, you can reach a high grade.",
            "priority": "medium"
        })
    
    # G1 → G2 trend
    if student.G2 < student.G1:
        suggestions.append({
            "icon": "📉",
            "title": "Reverse the Downward Trend",
            "detail": f"Your grade dropped from G1={student.G1} to G2={student.G2}. Identify what changed and take corrective action now.",
            "priority": "high"
        })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return suggestions


# Pre-built topic explanations and YouTube search links
STUDY_TOPICS = {
    "algebra": {
        "explanation": "Algebra is the branch of mathematics dealing with symbols and rules for manipulating those symbols. It covers equations, inequalities, polynomials, and functions. Key concepts include solving linear equations (ax + b = 0), quadratic equations (ax² + bx + c = 0), factoring, and working with variables.",
        "subtopics": ["Linear Equations", "Quadratic Equations", "Polynomials", "Inequalities", "Functions"]
    },
    "geometry": {
        "explanation": "Geometry studies shapes, sizes, positions, and properties of space. It covers points, lines, angles, surfaces, and solids. Key areas include triangles, circles, area/perimeter calculations, the Pythagorean theorem, and coordinate geometry.",
        "subtopics": ["Triangles", "Circles", "Area and Perimeter", "Pythagorean Theorem", "Coordinate Geometry"]
    },
    "calculus": {
        "explanation": "Calculus is the study of change and motion. It has two main branches: Differential Calculus (rates of change, derivatives) and Integral Calculus (accumulation, areas under curves). Key concepts include limits, derivatives, integrals, and the Fundamental Theorem of Calculus.",
        "subtopics": ["Limits", "Derivatives", "Integrals", "Chain Rule", "Applications"]
    },
    "statistics": {
        "explanation": "Statistics deals with collecting, analyzing, interpreting, and presenting data. Key concepts include mean, median, mode, standard deviation, probability distributions, hypothesis testing, and regression analysis.",
        "subtopics": ["Mean, Median, Mode", "Probability", "Normal Distribution", "Hypothesis Testing", "Regression"]
    },
    "trigonometry": {
        "explanation": "Trigonometry studies relationships between angles and sides of triangles. Core functions are sine, cosine, and tangent. It's essential for physics, engineering, and advanced math. Key topics include the unit circle, trigonometric identities, and solving triangles.",
        "subtopics": ["Sin, Cos, Tan", "Unit Circle", "Trigonometric Identities", "Inverse Functions", "Applications"]
    },
    "physics": {
        "explanation": "Physics is the science of matter, energy, and their interactions. Key areas include mechanics (forces, motion), thermodynamics (heat), electromagnetism, waves, and optics. Problem-solving often involves applying mathematical formulas to real-world scenarios.",
        "subtopics": ["Newton's Laws", "Kinematics", "Energy & Work", "Electricity", "Waves"]
    },
    "chemistry": {
        "explanation": "Chemistry studies matter, its properties, composition, and transformations. Key areas include atomic structure, chemical bonding, reactions, stoichiometry, acids/bases, and organic chemistry.",
        "subtopics": ["Atomic Structure", "Chemical Bonding", "Stoichiometry", "Acids & Bases", "Organic Chemistry"]
    },
    "biology": {
        "explanation": "Biology is the study of living organisms and their processes. It covers cell biology, genetics, evolution, ecology, and human anatomy. Understanding biology involves both memorization of terms and understanding of processes.",
        "subtopics": ["Cell Biology", "Genetics", "Evolution", "Ecology", "Human Anatomy"]
    },
    "programming": {
        "explanation": "Programming involves writing instructions for computers using languages like Python, Java, or C++. Key concepts include variables, data types, loops, conditionals, functions, and object-oriented programming (OOP).",
        "subtopics": ["Python Basics", "Data Structures", "Algorithms", "OOP", "Web Development"]
    },
    "english": {
        "explanation": "English studies cover reading comprehension, grammar, writing skills, and literature analysis. Key areas include essay writing, grammar rules, vocabulary building, and critical analysis of texts.",
        "subtopics": ["Grammar", "Essay Writing", "Reading Comprehension", "Vocabulary", "Literature Analysis"]
    },
    "mathematics": {
        "explanation": "Mathematics is the study of numbers, quantities, shapes, and patterns. It encompasses arithmetic, algebra, geometry, calculus, and statistics. Strong math skills require understanding concepts and regular practice.",
        "subtopics": ["Arithmetic", "Algebra", "Geometry", "Calculus", "Statistics"]
    },
    "portuguese": {
        "explanation": "Portuguese language studies cover grammar rules, reading comprehension, writing, and oral communication. Key areas include verb conjugation, sentence structure, vocabulary, and text interpretation.",
        "subtopics": ["Grammar", "Verb Conjugation", "Reading Comprehension", "Writing", "Vocabulary"]
    }
}


class StudyHelpRequest(BaseModel):
    subject: str
    topic: str = ""


@app.post("/predict")
def predict(student: StudentData):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded. Run train_models.py first.")
    
    student_dict = student.model_dump()
    df = pd.DataFrame([student_dict])
    
    try:
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)
        predicted_grade = float(prediction[0])
        
        # Clamp to 0-20 range
        predicted_grade = max(0, min(20, predicted_grade))
        
        suggestions = generate_suggestions(student, predicted_grade)
        
        # Determine grade level
        if predicted_grade >= 16:
            grade_level = "Excellent"
            grade_color = "green"
        elif predicted_grade >= 14:
            grade_level = "Good"
            grade_color = "blue"
        elif predicted_grade >= 10:
            grade_level = "Satisfactory"
            grade_color = "yellow"
        else:
            grade_level = "Needs Improvement"
            grade_color = "red"
        
        return {
            "predicted_grade": round(predicted_grade, 1),
            "grade_level": grade_level,
            "grade_color": grade_color,
            "max_grade": 20,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/study-help")
def study_help(request: StudyHelpRequest):
    subject = request.subject.strip().lower()
    topic = request.topic.strip()
    
    # Build search query
    search_query = subject
    if topic:
        search_query += f" {topic}"
    
    # Generate YouTube links
    yt_search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(search_query + ' tutorial')}"
    
    youtube_links = [
        {
            "title": f"🔍 Search YouTube: \"{search_query} tutorial\"",
            "url": yt_search_url
        },
        {
            "title": f"📺 Khan Academy: \"{subject}\"",
            "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote('khan academy ' + search_query)}"
        },
        {
            "title": f"🎓 Crash Course: \"{subject}\"",
            "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote('crash course ' + search_query)}"
        },
        {
            "title": f"📝 Organic Chemistry Tutor: \"{search_query}\"",
            "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote('organic chemistry tutor ' + search_query)}"
        },
        {
            "title": f"💡 3Blue1Brown: \"{subject}\"",
            "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote('3blue1brown ' + search_query)}"
        }
    ]
    
    # Add web resources
    web_links = [
        {
            "title": f"📖 Khan Academy",
            "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={urllib.parse.quote(search_query)}"
        },
        {
            "title": f"📗 Wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(search_query.replace(' ', '_'))}"
        }
    ]
    
    # Find matching topic explanation
    explanation = None
    subtopics = []
    
    # Check exact match first
    if subject in STUDY_TOPICS:
        info = STUDY_TOPICS[subject]
        explanation = info["explanation"]
        subtopics = info["subtopics"]
    else:
        # Fuzzy match
        for key, info in STUDY_TOPICS.items():
            if key in subject or subject in key:
                explanation = info["explanation"]
                subtopics = info["subtopics"]
                break
    
    if not explanation:
        explanation = f"'{subject.title()}' is an important academic subject. To improve, focus on understanding the core concepts, practice regularly, and use the resources below. Breaking down the topic into smaller parts and studying consistently is more effective than cramming."
    
    return {
        "subject": subject.title(),
        "topic": topic if topic else "General",
        "explanation": explanation,
        "subtopics": subtopics,
        "youtube_links": youtube_links,
        "web_links": web_links
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
