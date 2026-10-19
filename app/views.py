
# Python modules
import random
import os
import re

# Flask modules
from flask import render_template, request, url_for, redirect, send_from_directory
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
# App modules
from app        import app, lm, db, bc
from app.models import Users, Tests, Students
from app.forms  import LoginForm, RegisterForm
from app.QA_Gen_Model import QA_Gen_Model
from app.Objective_QA_Gen_Model import MCQ_Generator
from app.QA_Score import QA_Score
UPLOAD_FOLDER = 'Uploaded Material'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# Logout user
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# Register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    
    # declare the Registration Form
    form = RegisterForm(request.form)

    msg     = None
    success = False

    if request.method == 'GET': 

        return render_template( 'register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        # filter User out of database through username
        user = Users.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = Users.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'
        
        else:         

            pw_hash = bc.generate_password_hash(password)

            user = Users(username, email, pw_hash)

            user.save()

            msg     = 'User created, please <a href="' + url_for('login') + '">login</a>'     
            success = True

    else:
        msg = 'Input error'     

    return render_template( 'register.html', form=form, msg=msg, success=success )

# Authenticate user
@app.route('/login', methods=['GET', 'POST'])
def login():
    
    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = Users.query.filter_by(user=username).first()

        if user:
            
            if bc.check_password_hash(user.password, password):
                login_user(user)
                if(username =="admin"):
                    return redirect(url_for('tutorDashboard'))
                else :
                    return redirect(url_for('studentDashboard'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unknown user"

    return render_template( 'login.html', form=form, msg=msg )

# App main route + generic routing
@app.route('/', methods=['GET', 'POST'])
def index():

    try:
        # Declare the login form
        loginForm = LoginForm(request.form)

        # Flask message injected into the page, in case of any errors
        loginMsg = None    
        # check if both http method is POST and form is valid on submit
        if loginForm.validate_on_submit():
       
            # assign form data to variables
            username = request.form.get('username', '', type=str)
            password = request.form.get('password', '', type=str) 
         
            # filter User out of database through username
            user = Users.query.filter_by(user=username).first()

            if user:
            
                if bc.check_password_hash(user.password, password):
                    login_user(user)
                    if(username =="admin"):
                        return redirect(url_for('tutorDashboard'))
                    else :
                        return redirect(url_for('studentDashboard'))
                else:
                    loginMsg = "Wrong password. Please try again."
            else:
                loginMsg = "Unknown user"
        
         # declare the Registration Form
        registerForm = RegisterForm(request.form)

        registerMsg     = None
        success = False

        if registerForm.validate_on_submit():

            # assign form data to variables
            username = request.form.get('username', '', type=str)
            password = request.form.get('password', '', type=str) 
            email    = request.form.get('email'   , '', type=str) 

            # filter User out of database through username
            user = Users.query.filter_by(user=username).first()

            # filter User out of database through username
            user_by_email = Users.query.filter_by(email=email).first()

            if user or user_by_email:
                registerMsg = 'Error: User exists!'
        
            else:         

                pw_hash = bc.generate_password_hash(password)

                user = Users(username, email, pw_hash)

                user.save()

                registerMsg     = 'User created, please <a href="' + url_for('login') + '">login</a>'     
                success = True

        else:
            registerMsg = 'Input error'     

        return render_template( 'index.html', loginForm=loginForm, loginMsg=loginMsg, registerForm=registerForm, registerMsg=registerMsg)

    
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
#Render Student Dashboard
@app.route('/studentDashboard')
@login_required
def studentDashboard():
    tests = Tests.query.all()
    temp_tests = []
    for data in range(0,len(tests)):
        temp_tests.append([tests[data].id, tests[data].course_name, tests[data].test_name,"Objective", False])
        temp_tests.append([tests[data].id, tests[data].course_name, tests[data].test_name, "Subjective", False])
    students = Students.query.all()
    for data in range(0,len(temp_tests)):
        for ele in range(0,len(students)):
            if students[ele].course_id == temp_tests[data][0] and students[ele].test_name == temp_tests[data][2] and students[ele].type == temp_tests[data][3]:
                temp_tests[data][4]=True
                
    if current_user.user == "admin":
        return render_template('page-404.html'), 404
    else :
        return render_template( 'studentDashboard.html',name = current_user.user , email= current_user.email, temp_tests=temp_tests, students = students, tests=tests)

#Render Tutor Dashboard
@app.route('/tutorDashboard', methods=['GET', 'POST'])
@login_required
def tutorDashboard():
    tests = Tests.query.all()
    name = current_user.user
    if name=="admin" :               
        return render_template( 'tutorDashboard.html',name = name , email= current_user.email, tests = tests)
    else :
        return render_template('page-404.html'), 404


@app.route('/Subjective_Questions/<int:id>', methods=["GET","POST"])
@login_required
def Subjective_QA_Generation(id): 
    name = current_user.user
    data = Tests.query.get(id)
    response = []
    if request.method == "POST" :
            for i in range(3):
                response.append(request.form.get(str(i)))
            data = Tests.query.get(id)
            path = os.path.join(app.config['UPLOAD_FOLDER'], data.file_name)
            with open(path, 'r') as f:
                content = f.read()
            
            score = QA_Score(content,response)
            s = Students(course_id=id, course_name=data.course_name, test_name=data.test_name, type = "Subjective", score=score)
            db.session.add(s)
            db.session.commit()
            return redirect(url_for('responses', score=score, id=id))
    else :    
        if name!="admin" :
            path = os.path.join(app.config['UPLOAD_FOLDER'], data.file_name)
            with open(path, 'r') as f:
                content = f.read()
            que, ans = QA_Gen_Model.generate_test(content)
            size = len(que)
            
            return render_template('Subjective_Questions.html', question = que, answer = ans, size = size , id=id)
        else :
            return render_template('unAuth.html')
    

#Objective Question Generation
@app.route('/Objective_QA_Generation/<int:id>')
@login_required 
def Objective_QA_Generation(id):
    name = current_user.user
    data = Tests.query.get(id)
    if name!="admin" :
        path = os.path.join(app.config['UPLOAD_FOLDER'], data.file_name)
        with open(path, 'r') as f:
            content = f.read()
        Objective_Questions = MCQ_Generator.generate_mcq_questions(content, 9)
        answers = []
        for questions in Objective_Questions:
            questions.distractors.append(questions.answerText)
            random.shuffle(questions.distractors)
            answers.append(questions.answerText)
        return render_template('Objective_Questions.html', Objective_Questions = Objective_Questions, answers = answers,id=id)
    else :
        return render_template('unAuth.html')



#responses
@app.route('/response/<int:id>/<int:score>')
@login_required
def response(id,score):
    name = current_user.user
    data = Tests.query.get(id) 
    s = Students(course_id=id, course_name=data.course_name, test_name=data.test_name, type = "Objective", score=score)
    db.session.add(s)
    db.session.commit()
    if name !="admin":
        return render_template('response.html', score=score, name = name, data= data)
    else :
        return render_template('unAuth.html')

#responses
@app.route('/responses/<int:score>/<int:id>')
@login_required
def responses(score, id):
    name = current_user.user
    data = Tests.query.get(id)
    if name !="admin":
        return render_template('response.html', name = name, score = score, data = data)
    else :
        return render_template('unAuth.html')


@app.route('/add', methods=["POST"])
@login_required
def add():
    # In this function we will input data from the
    # form page and store it in our database. Remember
    # that inside the get the name should exactly be the same
    # as that in the html input fields
    
    course_name = request.form.get("course_name")
    test_name = request.form.get("test_name")
    file = request.files['file']
    file_name = secure_filename(file.filename)
    if file_name != '':
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

    # create an object of the course class of models and
    # store data as a row in our datatable
    if test_name != '' and course_name != '' :
        t = Tests(course_name=course_name, test_name=test_name, file_name=file_name)
        db.session.add(t)
        db.session.commit()
        return redirect('/tutorDashboard')
    else:
        return redirect('/tutorDashboard')

@app.route('/delete/<int:id>')
@login_required
def erase(id):
     
    # deletes the data on the basis of unique id 
    data = Tests.query.get(id)
    db.session.delete(data)
    db.session.commit()
    return redirect('/tutorDashboard')

@app.route('/dlt/<int:id>')
@login_required
def dlt(id):
    # deletes the data on the basis of unique id 
    data = Students.query.get(id)
    db.session.delete(data)
    db.session.commit()
    return redirect('/studentDashboard')

# Return sitemap
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')
