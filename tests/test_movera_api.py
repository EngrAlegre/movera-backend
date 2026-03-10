"""
MovEra API Backend Tests
Tests cover: Auth, Profile, Onboarding, Meal Plans, Workout Plans, AI Coach, Daily Logs, Dashboard, Stats
"""
import pytest
import requests
import os
import time

# Get backend URL from environment
BACKEND_URL = os.environ.get('EXPO_PUBLIC_BACKEND_URL')
if not BACKEND_URL:
    raise ValueError("EXPO_PUBLIC_BACKEND_URL not set")

BASE_URL = BACKEND_URL.rstrip('/')

# Test user credentials
TEST_EMAIL = f"test_user_{int(time.time())}@movera.test"
TEST_PASSWORD = "TestPass123!"
TEST_NAME = "Test User"


class TestAuth:
    """Authentication endpoints"""
    
    def test_register_success(self):
        """Test user registration"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "name": TEST_NAME
        })
        print(f"Register response: {response.status_code}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "token" in data, "Token not in response"
        assert "user" in data, "User not in response"
        assert data["user"]["email"] == TEST_EMAIL.lower()
        assert data["user"]["name"] == TEST_NAME
        assert data["user"]["onboarding_completed"] == False
        
        # Store token for other tests
        pytest.token = data["token"]
        pytest.user_id = data["user"]["id"]
        print(f"✓ User registered successfully with ID: {pytest.user_id}")
    
    def test_register_duplicate_email(self):
        """Test duplicate email registration fails"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "name": TEST_NAME
        })
        assert response.status_code == 400
        print("✓ Duplicate email rejected")
    
    def test_login_success(self):
        """Test login with correct credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "user" in data
        print("✓ Login successful")
    
    def test_login_wrong_password(self):
        """Test login with wrong password"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": "WrongPassword123"
        })
        assert response.status_code == 401
        print("✓ Wrong password rejected")
    
    def test_get_me_authenticated(self):
        """Test /auth/me with valid token"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/auth/me", 
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == TEST_EMAIL.lower()
        print("✓ GET /auth/me works")
    
    def test_get_me_no_token(self):
        """Test /auth/me without token"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401
        print("✓ No token rejected")


class TestOnboarding:
    """Profile and onboarding endpoints"""
    
    def test_complete_onboarding(self):
        """Test completing onboarding"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/profile/onboarding",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={
                "name": "Updated Test User",
                "age": 28,
                "weight": 75.5,
                "weight_unit": "kg",
                "height": 175.0,
                "height_unit": "cm",
                "fitness_goal": "Build Muscle",
                "activity_level": "Active",
                "experience_level": "Intermediate",
                "workout_style": "Gym",
                "lifestyle_mode": "Full Access"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["onboarding_completed"] == True
        assert data["name"] == "Updated Test User"
        assert data["age"] == 28
        assert data["fitness_goal"] == "Build Muscle"
        print("✓ Onboarding completed")
    
    def test_get_profile(self):
        """Test GET /profile"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/profile",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "name" in data
        print("✓ GET /profile works")
    
    def test_update_profile(self):
        """Test PUT /profile"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.put(f"{BASE_URL}/api/profile",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"age": 29, "fitness_goal": "Lose Weight"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["age"] == 29
        assert data["fitness_goal"] == "Lose Weight"
        print("✓ Profile updated")


class TestMealPlans:
    """Meal plan generation and management"""
    
    def test_generate_meal_plan_7days(self):
        """Test generating 7-day meal plan (fallback mode)"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/meals/generate",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"days": 7}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "plan_data" in data
        assert data["days"] == 7
        assert len(data["plan_data"]) == 7
        
        # Check meal structure
        day_one = data["plan_data"][0]
        assert "meals" in day_one
        assert "breakfast" in day_one["meals"]
        assert "lunch" in day_one["meals"]
        assert "dinner" in day_one["meals"]
        
        # Store plan ID
        pytest.meal_plan_id = data["id"]
        print(f"✓ 7-day meal plan generated with ID: {pytest.meal_plan_id}")
    
    def test_get_current_meal_plan(self):
        """Test GET /meals/current"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/meals/current",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        if data:  # Plan exists
            assert "id" in data
            assert "plan_data" in data
            print("✓ Current meal plan retrieved")
        else:
            print("✓ No meal plan exists (expected for new user)")
    
    def test_mark_meal_eaten(self):
        """Test marking a meal as eaten"""
        if not hasattr(pytest, 'token') or not hasattr(pytest, 'meal_plan_id'):
            pytest.skip("No token or meal plan available")
        
        response = requests.post(f"{BASE_URL}/api/meals/eaten",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={
                "plan_id": pytest.meal_plan_id,
                "day_index": 0,
                "meal_type": "breakfast"
            }
        )
        assert response.status_code == 200
        print("✓ Meal marked as eaten")


class TestWorkoutPlans:
    """Workout plan generation and management"""
    
    def test_generate_workout_plan_7days(self):
        """Test generating 7-day workout plan (fallback mode)"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/workouts/generate",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"days": 7}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "plan_data" in data
        assert data["days"] == 7
        assert len(data["plan_data"]) == 7
        
        # Check workout structure
        day_one = data["plan_data"][0]
        assert "workout_type" in day_one
        assert "exercises" in day_one
        assert len(day_one["exercises"]) > 0
        
        # Store plan ID
        pytest.workout_plan_id = data["id"]
        print(f"✓ 7-day workout plan generated with ID: {pytest.workout_plan_id}")
    
    def test_get_current_workout_plan(self):
        """Test GET /workouts/current"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/workouts/current",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        if data:
            assert "id" in data
            assert "plan_data" in data
            print("✓ Current workout plan retrieved")
    
    def test_mark_workout_done(self):
        """Test marking workout as completed"""
        if not hasattr(pytest, 'token') or not hasattr(pytest, 'workout_plan_id'):
            pytest.skip("No token or workout plan available")
        
        response = requests.post(f"{BASE_URL}/api/workouts/complete",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={
                "plan_id": pytest.workout_plan_id,
                "day_index": 0
            }
        )
        assert response.status_code == 200
        print("✓ Workout marked as complete")
    
    def test_submit_workout_feedback(self):
        """Test submitting workout feedback"""
        if not hasattr(pytest, 'token') or not hasattr(pytest, 'workout_plan_id'):
            pytest.skip("No token or workout plan available")
        
        response = requests.post(f"{BASE_URL}/api/workouts/feedback",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={
                "plan_id": pytest.workout_plan_id,
                "day_index": 0,
                "feedback": "Too Easy"
            }
        )
        assert response.status_code == 200
        print("✓ Workout feedback submitted")


class TestDailyLogs:
    """Daily logging endpoints"""
    
    def test_log_water(self):
        """Test logging water intake"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/logs/water",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"amount_ml": 500}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["water_ml"] >= 500
        print("✓ Water logged")
    
    def test_log_steps(self):
        """Test logging steps"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/logs/steps",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"steps": 5000}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["steps"] == 5000
        print("✓ Steps logged")
    
    def test_get_daily_log(self):
        """Test GET /logs/daily"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/logs/daily",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert "date" in data
        assert "water_ml" in data
        assert "steps" in data
        print("✓ Daily log retrieved")
    
    def test_log_weight(self):
        """Test logging weight"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/logs/weight",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"weight": 76.0, "unit": "kg"}
        )
        assert response.status_code == 200
        print("✓ Weight logged")
    
    def test_get_weight_history(self):
        """Test GET /logs/weight-history"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/logs/weight-history",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print("✓ Weight history retrieved")


class TestDashboard:
    """Dashboard and stats endpoints"""
    
    def test_get_dashboard(self):
        """Test GET /dashboard"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/dashboard",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert "greeting_name" in data
        assert "calories_goal" in data
        assert "water_ml" in data
        assert "steps" in data
        assert "workout_status" in data
        assert "quote" in data
        assert "workouts_this_week" in data
        assert "streak" in data
        assert "has_meal_plan" in data
        assert "has_workout_plan" in data
        print("✓ Dashboard data retrieved")
    
    def test_get_stats(self):
        """Test GET /stats"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/stats",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert "total_workouts" in data
        assert "total_meal_plans" in data
        assert "current_streak" in data
        assert "achievements" in data
        print("✓ Stats retrieved")
    
    def test_get_achievements(self):
        """Test GET /achievements"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/achievements",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Achievements retrieved: {len(data)} badges")


class TestAICoach:
    """AI Coach chat endpoints (fallback mode - will fail without Perplexity key)"""
    
    def test_chat_without_perplexity_key(self):
        """Test AI chat fails gracefully without API key"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/ai/chat",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"message": "Hello Era, how can I lose weight?"}
        )
        # Should return 500 since PERPLEXITY_API_KEY is not set
        assert response.status_code == 500
        data = response.json()
        assert "Perplexity API key not configured" in data.get("detail", "")
        print("✓ AI chat correctly fails without API key (expected)")
    
    def test_get_chat_history(self):
        """Test GET /ai/history"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.get(f"{BASE_URL}/api/ai/history",
                                headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print("✓ Chat history retrieved")
    
    def test_clear_chat_history(self):
        """Test DELETE /ai/history"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.delete(f"{BASE_URL}/api/ai/history",
                                   headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        print("✓ Chat history cleared")


class TestReflections:
    """Daily reflections endpoints"""
    
    def test_submit_reflection(self):
        """Test POST /reflections"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.post(f"{BASE_URL}/api/reflections",
            headers={"Authorization": f"Bearer {pytest.token}"},
            json={"mood": "happy", "note": "Great workout today!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "reflection" in data
        print("✓ Reflection submitted")


# Cleanup test
class TestCleanup:
    """Cleanup test data"""
    
    def test_delete_account(self):
        """Test DELETE /account (cleanup)"""
        if not hasattr(pytest, 'token'):
            pytest.skip("No token available")
        
        response = requests.delete(f"{BASE_URL}/api/account",
                                   headers={"Authorization": f"Bearer {pytest.token}"})
        assert response.status_code == 200
        print("✓ Test account deleted (cleanup)")
