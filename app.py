from flask import Flask, render_template, request, jsonify
import joblib # For loading the ML model
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


# Blood test reference information
BLOOD_TESTS = {
    'Hemoglobin': {
        'name': 'Hemoglobin',
        'units': 'g/dL',
        'ranges': {
            'male': {'min': 13.5, 'max': 17.5},
            'female': {'min': 12.0, 'max': 15.5}
        },
        'low': {
            'condition': 'Anemia, Blood loss, Chronic disease, Nutritional deficiency, Bone marrow disorder, Kidney disease',
            'symptoms': 'Fatigue, Weakness, Pale skin, Shortness of breath, Dizziness'
        },
        'high': {
            'condition': 'Dehydration, Polycythemia vera, Lung disease, High altitude adaptation',
            'symptoms': 'Headache, Dizziness, Flushed skin, Blurred vision, Itching'
        }
    },
    'RBC': {
        'name': 'Red Blood Cells',
        'units': 'million cells/μL',
        'ranges': {
            'male': {'min': 4.5, 'max': 5.9},
            'female': {'min': 4.0, 'max': 5.2}
        },
        'low': {
            'condition': 'Anemia, Bone marrow failure, Nutritional deficiency, Chronic inflammation, Hemolysis',
            'symptoms': 'Fatigue, Pale skin, Rapid heartbeat, Cold hands/feet'
        },
        'high': {
            'condition': 'Dehydration, Polycythemia vera, Hypoxia, Kidney tumor',
            'symptoms': 'Fatigue, Headache, Blurred vision, Itching (especially after shower)'
        }
    },
    'HCT': {
        'name': 'Hematocrit',
        'units': '%',
        'ranges': {
            'male': {'min': 40, 'max': 50},
            'female': {'min': 36, 'max': 46}
        },
        'low': {
            'condition': 'Anemia, Bleeding, Nutritional deficiency, Bone marrow disorder',
            'symptoms': 'Fatigue, Weakness, Pale skin, Shortness of breath'
        },
        'high': {
            'condition': 'Dehydration, Polycythemia vera, Chronic lung disease',
            'symptoms': 'Headache, Dizziness, Flushed skin, Vision problems'
        }
    },
    'MCV': {
        'name': 'Mean Corpuscular Volume',
        'units': 'fL',
        'ranges': {'min': 80, 'max': 100},
        'low': {
            'condition': 'Iron deficiency anemia, Thalassemia, Chronic disease',
            'symptoms': 'Fatigue, Pale skin, Brittle nails, Pica (craving ice/dirt)'
        },
        'high': {
            'condition': 'Vitamin B12 deficiency, Folate deficiency, Liver disease, Hypothyroidism',
            'symptoms': 'Fatigue, Diarrhea, Numbness/tingling, Balance problems'
        }
    },
    'MCH': {
        'name': 'Mean Corpuscular Hemoglobin',
        'units': 'pg',
        'ranges': {'min': 27, 'max': 33},
        'low': {
            'condition': 'Iron deficiency anemia, Thalassemia',
            'symptoms': 'Fatigue, Pale skin, Weakness, Shortness of breath'
        },
        'high': {
            'condition': 'Macrocytic anemia, Reticulocytosis',
            'symptoms': 'Fatigue, Pale skin, Diarrhea, Numbness in extremities'
        }
    },
    'MCHC': {
        'name': 'Mean Corpuscular Hemoglobin Concentration',
        'units': 'g/dL',
        'ranges': {'min': 32, 'max': 36},
        'low': {
            'condition': 'Iron deficiency anemia, Thalassemia',
            'symptoms': 'Fatigue, Pale skin, Brittle nails, Cold intolerance'
        },
        'high': {
            'condition': 'Hereditary spherocytosis, Hemoglobin C disease',
            'symptoms': 'Fatigue, Jaundice, Enlarged spleen, Gallstones'
        }
    },
    'RDW-CV': {
        'name': 'Red Cell Distribution Width (CV)',
        'units': '%',
        'ranges': {'min': 11.5, 'max': 14.5},
        'low': {
            'condition': 'Not clinically significant',
            'symptoms': 'None typically'
        },
        'high': {
            'condition': 'Iron deficiency anemia, Vitamin B12 deficiency, Hemoglobinopathy, Myelodysplasia',
            'symptoms': 'Varies by underlying condition (fatigue, weakness, pallor)'
        }
    },
    'RDW-SD': {
        'name': 'Red Cell Distribution Width (SD)',
        'units': 'fL',
        'ranges': {'min': 39, 'max': 46},
        'low': {
            'condition': 'Not clinically significant',
            'symptoms': 'None typically'
        },
        'high': {
            'condition': 'Iron deficiency anemia, Vitamin B12 deficiency, Hemoglobinopathy, Myelodysplasia',
            'symptoms': 'Varies by underlying condition (fatigue, weakness, pallor)'
        }
    },
    'WBC': {
        'name': 'White Blood Cells',
        'units': '×10³/μL',
        'ranges': {'min': 4.0, 'max': 11.0},
        'low': {
            'condition': 'Viral infection, Bone marrow disorder, Autoimmune disease, Severe infection',
            'symptoms': 'Frequent infections, Fever, Fatigue, Mouth sores'
        },
        'high': {
            'condition': 'Bacterial infection, Leukemia, Inflammation, Stress response',
            'symptoms': 'Fever, Pain, Fatigue, Night sweats (if leukemia)'
        }
    },
    'NEU%': {
        'name': 'Neutrophils',
        'units': '%',
        'ranges': {'min': 40, 'max': 70},
        'low': {
            'condition': 'Viral infection, Autoimmune disorder, Chemotherapy effect',
            'symptoms': 'Frequent infections, Fever, Mouth ulcers'
        },
        'high': {
            'condition': 'Bacterial infection, Acute inflammation, Steroid use',
            'symptoms': 'Fever, Pain, Redness/swelling at infection site'
        }
    },
    'LYM%': {
        'name': 'Lymphocytes',
        'units': '%',
        'ranges': {'min': 20, 'max': 40},
        'low': {
            'condition': 'HIV/AIDS, Immunosuppression, Radiation exposure',
            'symptoms': 'Frequent infections, Weight loss, Fatigue'
        },
        'high': {
            'condition': 'Viral infection, Chronic infection, Lymphoma',
            'symptoms': 'Swollen lymph nodes, Fever, Night sweats'
        }
    },
    'MON%': {
        'name': 'MON%',
        'units': '%',
        'ranges': {'min': 2, 'max': 10},
        'low': {
            'condition': '',
            'symptoms': ''
        },
        'high': {
            'condition': 'Chronic infection, Autoimmune disease, Myeloproliferative disorder',
            'symptoms': ''
        }
    },
    'EOS%': {
        'name': 'EOS%',
        'units': '%',
        'ranges': {'min': 0, 'max': 6},
        'low': {
            'condition': '',
            'symptoms': ''
        },
        'high': {
            'condition': 'Allergic disorder, Parasitic infection, Autoimmune disease',
            'symptoms': ''
        }
    },
    'BAS%': {
        'name': 'BAS%',
        'units': '%',
        'ranges': {'min': 0, 'max': 2},
        'low': {
            'condition': '',
            'symptoms': ''
        },
        'high': {
            'condition': 'Allergic reaction, Chronic inflammation, Myeloproliferative disorder',
            'symptoms': ''
        }
    },
    'LYM#': {
        'name': 'LYM#',
        'units': '×10³/μL',
        'ranges': {'min': 1.0, 'max': 4.0},
        'low': {
            'condition': 'HIV/AIDS, Immunosuppression',
            'symptoms': 'Frequent infections, Weight loss, Fatigue'
        },
        'high': {
            'condition': 'Viral infection, Lymphoma',
            'symptoms': 'Swollen lymph nodes, Fever, Night sweats'
        }
    },
    'GRA#': {
        'name': 'GRA#',
        'units': '×10³/μL',
        'ranges': {'min': 1.8, 'max': 7.0},
        'low': {
            'condition': 'Chemotherapy effect, Bone marrow failure',
            'symptoms': ''
        },
        'high': {
            'condition': 'Bacterial infection, Inflammation',
            'symptoms': ''
        }
    },
    'PLT': {
        'name': 'Platelets',
        'units': '×10³/μL',
        'ranges': {'min': 150, 'max': 450},
        'low': {
            'condition': 'Viral infection, Autoimmune disorder, Bone marrow disorder',
            'symptoms': 'Easy bruising, Prolonged bleeding, Petechiae (small red spots)'
        },
        'high': {
            'condition': 'Inflammation, Iron deficiency, Myeloproliferative disorder',
            'symptoms': 'Headache, Dizziness, Blood clots (in extreme cases)'
        }
    },
    'ESR': {
        'name': 'Erythrocyte Sedimentation Rate',
        'units': 'mm/hr',
        'ranges': {
            'male': {'min': 0, 'max': 15},
            'female': {'min': 0, 'max': 20}
        },
        'low': {
            'condition': 'Not clinically significant',
            'symptoms': 'None'
        },
        'high': {
            'condition': 'Inflammation, Infection, Autoimmune disease, Malignancy',
            'symptoms': 'Depends on underlying condition (joint pain, fever, fatigue)'
        }
    }
}

# Rule-based recommendations for common blood test abnormalities
# This dictionary contains conditions and their respective recommendations
# The keys are the names of the conditions, and the values are dictionaries with descriptions and recommendations
# Each recommendation is a list of actionable items for the user to follow
ABNORMALITIES = {
    'Allergic reaction': {
        'description': 'An immune system response to a foreign substance that is typically harmless to most people.',
        'recommendations': [
            'Identify and avoid known allergens',
            'Use antihistamines for mild reactions',
            'Carry epinephrine auto-injector if severe allergies exist',
            'Consult an allergist for testing and management'
        ]
    },
    'Anemia': {
        'description': 'Low hemoglobin may indicate anemia, which can be caused by iron deficiency, vitamin B12 deficiency, chronic disease, or blood loss.',
        'recommendations': [
            'Increase iron-rich foods (red meat, spinach, lentils)',
            'Consume vitamin C to enhance iron absorption',
            'Consider iron supplements if deficient',
            'Consult doctor if symptoms persist'
        ]
    },
    'Autoimmune disease': {
        'description': 'A condition where the immune system mistakenly attacks the body\'s own tissues.',
        'recommendations': [
            'Consult a rheumatologist for proper diagnosis',
            'Monitor for worsening symptoms',
            'Follow prescribed treatment plan',
            'Maintain regular follow-up appointments'
        ]
    },
    'Bacterial infection': {
        'description': 'An infection caused by harmful bacteria multiplying in the body.',
        'recommendations': [
            'Complete prescribed antibiotic course',
            'Increase fluid intake',
            'Get adequate rest',
            'Monitor for fever or worsening symptoms'
        ]
    },
    'Blood loss': {
        'description': 'Reduction in blood volume, potentially leading to anemia.',
        'recommendations': [
            'Identify and address source of bleeding',
            'Increase iron-rich foods to support recovery',
            'Monitor for signs of anemia (fatigue, pallor)',
            'Seek medical attention for significant blood loss'
        ]
    },
    'Bone marrow disorder': {
        'description': 'Conditions affecting blood cell production in the bone marrow.',
        'recommendations': [
            'Consult a hematologist for evaluation',
            'Monitor blood counts regularly',
            'Avoid activities that may cause bleeding',
            'Follow recommended treatment plan'
        ]
    },
    'Chronic inflammation': {
        'description': 'Long-term inflammatory response that can damage tissues.',
        'recommendations': [
            'Follow anti-inflammatory diet (rich in fruits, vegetables, omega-3s)',
            'Maintain healthy weight',
            'Exercise regularly',
            'Manage stress through relaxation techniques'
        ]
    },
    'Hemolysis': {
        'description': 'Premature destruction of red blood cells leading to anemia.',
        'recommendations': [
            'Identify and treat underlying cause',
            'Monitor for jaundice or dark urine',
            'Increase folic acid intake',
            'Consult hematologist for evaluation'
        ]
    },
    'Iron deficiency': {
        'description': 'Insufficient iron stores to meet the body\'s needs.',
        'recommendations': [
            'Increase dietary iron intake (red meat, beans, fortified cereals)',
            'Combine iron-rich foods with vitamin C sources',
            'Avoid tea/coffee with meals (can inhibit iron absorption)',
            'Consider iron supplements if recommended by doctor'
        ]
    },
    'Iron deficiency anemia': {
        'description': 'Anemia resulting from insufficient iron to produce hemoglobin.',
        'recommendations': [
            'Increase iron-rich foods in diet',
            'Take iron supplements as prescribed',
            'Retest hemoglobin after treatment period',
            'Investigate potential causes of iron loss'
        ]
    },
    'Kidney disease': {
        'description': 'Impaired kidney function affecting blood cell production.',
        'recommendations': [
            'Monitor kidney function tests',
            'Control blood pressure and diabetes if present',
            'Follow renal diet if recommended',
            'Consult nephrologist for management'
        ]
    },
    'Thalassemia': {
        'description': 'Genetic disorder causing abnormal hemoglobin production.',
        'recommendations': [
            'Consult hematologist for specialized care',
            'Monitor iron levels (avoid unnecessary supplements)',
            'Consider genetic counseling',
            'Maintain regular follow-up appointments'
        ]
    },
    'Vitamin B12 deficiency': {
        'description': 'Insufficient B12 affecting red blood cell production.',
        'recommendations': [
            'Increase animal product consumption (meat, eggs, dairy)',
            'Consider B12 supplements or injections',
            'Get tested for pernicious anemia',
            'Monitor neurological symptoms'
        ]
    },
    'Myeloproliferative disorder': {
        'description': 'Conditions causing overproduction of blood cells by bone marrow.',
        'recommendations': [
            'Consult hematologist for specialized care',
            'Monitor blood counts regularly',
            'Follow prescribed treatment plan',
            'Report any unusual bleeding or clotting'
        ]
    },
    'Lymphoma': {
        'description': 'Cancer of the lymphatic system affecting blood cells.',
        'recommendations': [
            'Consult oncologist for evaluation',
            'Follow recommended diagnostic tests',
            'Discuss treatment options with specialist',
            'Seek support from cancer care team'
        ]
    },
    'Chronic infection': {
        'description': 'Persistent infection affecting blood cell production.',
        'recommendations': [
            'Identify and treat underlying infection',
            'Support immune system with balanced nutrition',
            'Get adequate rest',
            'Follow prescribed treatment regimen'
        ]
    }
}


# Load ML model and related files
try:
    model = joblib.load('static/ml_model/blood_report_model.pkl') # Load the trained ML model
    mlb = joblib.load('static/ml_model/label_binarizer.pkl') # Load the label binarizer for multi-label classification
    imputer = joblib.load('static/ml_model/imputer.pkl') # Load the imputer for missing values
    train_cols = joblib.load('static/ml_model/training_columns.pkl') # Load the training columns to match the input data
    print(train_cols)

    print("ML model and dependencies loaded successfully")

except Exception as e:
    print(f"Error loading ML model files: {e}")
    model = None


@app.route('/')
def home():
    return render_template('index.html')

# This route is for the enter report page
@app.route('/enter-report')
def enter_report():
    return render_template('enter_report.html', tests=BLOOD_TESTS)

# This route is for the results page
@app.route('/results')
def results():
    return render_template('results.html')


# Route for handling the API request to analyze blood test results
@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if not data or 'gender' not in data or 'age' not in data or 'testResults' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    gender = data['gender'].lower()
    age = int(data['age'])
    test_results = data['testResults']
    
    analysis = {}
    abnormalities = []
    ml_input = {'Age': age, 'Sex': gender}
    print(test_results)
    
    # Rule based analysis i.e if the value is out of range, then check the condition and symptoms
    for test_key, value in test_results.items():
        test_info = BLOOD_TESTS.get(test_key)
        if not test_info or value is None:
            continue
            
        # Rule-based analysis
        ref_range = test_info['ranges'].get(gender, test_info['ranges'])

        # print(ref_range)
        
        if ref_range:
            status = 'normal'
            condition = symptoms = None
            
            if value < ref_range['min']:
                status = 'low'
                condition = test_info.get('low', {}).get('condition')
                symptoms = test_info.get('low', {}).get('symptoms')
            elif value > ref_range['max']:
                status = 'high'
                condition = test_info.get('high', {}).get('condition')
                symptoms = test_info.get('high', {}).get('symptoms')
            
            analysis[test_key] = {
                'name': test_info['name'],
                'value': value,
                'status': status,
                'units': test_info['units'],
                'reference_range': f"{ref_range['min']}-{ref_range['max']}",
                'condition': condition,
                'symptoms': symptoms
            }
            
            # if status != 'Normal':
            #     abnormalities.append({
            #         'test': test_info['name'],
            #         'value': f"{value} {test_info['units']}",
            #         'status': status,
            #         'range': f"{ref_range['min']}-{ref_range['max']}",
            #         'condition': condition,
            #         'symptoms': symptoms
            #     })

    # print(analysis)

    # taking the input columns in the same order as the training data
    for col in train_cols:
        if col in test_results:
            ml_input[col] = test_results[col]
    
    print(ml_input)

    # ML model prediction
    ml_predictions = predict_abnormalities(ml_input)

    # Health summary generation (but not implemented in Frontend yet)
    summary = generate_health_summary(analysis, gender, age)
    
    # recommendations = generate_range_recommendations(analysis)

    # Rule based recommendations
    recommendations = generate_recommendations(ml_predictions)
    
    report_data = {
        'gender': gender,
        'age': age,
        'analysis': analysis,
        'abnormalities': abnormalities,
        'ml_predictions': ml_predictions,
        'recommendations': recommendations,
        'summary': summary
    }
    
    # session['report_data'] = report_data
    
    return jsonify(report_data)


# Preprocessing the given data as per the model requirements and predict the abnormalities
def predict_abnormalities(patient_data):
    if model is None:
        return {'Error': 'ML model not loaded'}
    
    try:
        patient_df = pd.DataFrame([patient_data])
        num_cols = patient_df.select_dtypes(include=np.number).columns
        # print(train_cols)
        # print(num_cols)
        patient_df[num_cols] = imputer.transform(patient_df[num_cols])
        patient_df = pd.get_dummies(patient_df, columns=['Sex'])
        
        missing_cols = set(train_cols) - set(patient_df.columns)
        for col in missing_cols:
            patient_df[col] = 0
        patient_df = patient_df[train_cols]
        # print(patient_df.columns)
        probs = model.predict_proba(patient_df)
        predictions = {}
        
        for i, condition in enumerate(mlb.classes_):
            prob = probs[i][0][1]
            if prob > 0.9:  # Adjust threshold as needed
                predictions[condition] = round(prob, 4)*100
        
        return predictions or {'Normal': 100}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'Error': 'Prediction failed'}
    

def generate_health_summary(analysis, gender, age):
    abnormalities = []
    
    for test_name, result in analysis.items():
        if result['status'] != 'normal':
            abnormalities.append({
                'test': test_name,
                'value': result['value'],
                'status': result['status'],
                'reference_range': result['reference_range']
            })
    
    summary = []
    
    if not abnormalities:
        summary.append("All your blood test results are within normal ranges.")
    else:
        summary.append(f"Your blood test shows {len(abnormalities)} abnormal value(s):")
        
        for ab in abnormalities:
            if ab['status'] == 'low':
                summary.append(f"- Low {ab['test']} ({ab['value']}, normal range: {ab['reference_range']})")
            else:
                summary.append(f"- High {ab['test']} ({ab['value']}, normal range: {ab['reference_range']})")
    
    # Add age/gender specific notes
    if age > 50:
        summary.append("\nNote: Some reference ranges may vary slightly for your age group.")
    
    return "\n".join(summary)


# Rule based recommendations table
def generate_recommendations(ml_predictions):
    recommendations = []
    conditions_found = set()
    
    if len(ml_predictions) == 0 or (len(ml_predictions) == 1 and 'Normal' in ml_predictions):
        recommendations.append({
            'title': 'General Health',
            'items': [
                'No specific recommendations needed as all values are normal',
                'Maintain a balanced diet and regular exercise'
            ]
        })
    else:
        for condition, prob in ml_predictions.items():
            if condition == 'Normal':
                continue
            if condition in ABNORMALITIES:
                condition_data = ABNORMALITIES.get(condition)
                recommendations.append({
                    'title': f"For {condition.title()}",
                    'condition': condition,
                    'confidence': prob,
                    'description': condition_data['description'],
                    'items': condition_data['recommendations']
                })
    
    return recommendations


if __name__ == '__main__':
    app.run(debug=True)
