from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn


app = FastAPI(title='Predict Math score')


model = joblib.load('model_3.pkl')
scaler = joblib.load('scaler_3.pkl')


class StudentDataShema(BaseModel):
    gender: str
    reading: int
    writing: int
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str


@app.post('/predict', response_model=dict)
async def predict(data: StudentDataShema):
    student_data = data.dict()

    new_gender = student_data.pop('gender')
    gender_0_1 = [
        1 if new_gender == 'male' else 0,
    ]

    new_race = student_data.pop('race_ethnicity')
    race_0_1 = [
        1 if new_race == 'group B' else 0,
        1 if new_race == 'group C' else 0,
        1 if new_race == 'group D' else 0,
        1 if new_race == 'group E' else 0,
    ]

    new_education = student_data.pop('parental_level_of_education')
    education_0_1 = [
        1 if new_education == "bachelor's degree" else 0,
        1 if new_education == 'high school' else 0,
        1 if new_education == "master's degree" else 0,
        1 if new_education == 'some college' else 0,
        1 if new_education == 'some high school' else 0,
        1 if new_education == "associate's degree" else 0,
    ]

    new_lunch = student_data.pop('lunch')
    lunch_0_1 = [
        1 if new_lunch == 'standard' else 0
    ]

    new_course = student_data.pop('test_preparation_course')
    course_0_1 = [
        1 if new_course == 'none' else 0
    ]

    features = gender_0_1 + list(student_data.values())  + race_0_1 + education_0_1 + lunch_0_1 + course_0_1
    scaler_data = scaler.transform([features])
    predict = model.predict(scaler_data)[0]
    return {'predict': round(predict, 2)}





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)


