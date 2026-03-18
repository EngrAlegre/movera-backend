from fastapi import FastAPI, APIRouter, Depends, HTTPException, Header
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
import logging
import json
import re
import httpx
import uuid
import hashlib
import bcrypt
import random
import string
import jwt as pyjwt
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Supabase
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_SERVICE_KEY = os.environ['SUPABASE_SERVICE_KEY']
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Config
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY', '').strip()
JWT_SECRET = os.environ.get('JWT_SECRET', 'movera-jwt-secret-2024-secure')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 168
PERPLEXITY_MODEL = 'sonar'

app = FastAPI(title="MovEra API")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== AUTH HELPERS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    payload = {
        'user_id': user_id,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc)
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def user_dict_from_row(row: dict) -> dict:
    """Convert a Supabase row to a user dict, excluding password_hash."""
    return {k: v for k, v in row.items() if k != 'password_hash'}

async def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Not authenticated')
    token = authorization.split(' ')[1]
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        result = supabase.table('users').select('*').eq('id', payload['user_id']).execute()
        if not result.data:
            raise HTTPException(status_code=401, detail='User not found')
        return result.data[0]
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token expired')
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid token')

# ==================== PERPLEXITY SERVICE ====================

async def call_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=500, detail='Perplexity API key not configured. Please set PERPLEXITY_API_KEY.')
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': PERPLEXITY_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': 0.7
    }
    async with httpx.AsyncClient(timeout=90.0) as http_client:
        response = await http_client.post(
            'https://api.perplexity.ai/chat/completions',
            json=payload, headers=headers
        )
        if response.status_code != 200:
            logger.error(f'Perplexity API error: {response.status_code} - {response.text}')
            raise HTTPException(status_code=502, detail=f'{response.status_code}: {response.text[:200]}')
        data = response.json()
        return data['choices'][0]['message']['content']

async def call_perplexity_chat(messages: list, max_tokens: int = 1500) -> str:
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=500, detail='Perplexity API key not configured. Please set PERPLEXITY_API_KEY.')
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': PERPLEXITY_MODEL,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': 0.7
    }
    async with httpx.AsyncClient(timeout=90.0) as http_client:
        response = await http_client.post(
            'https://api.perplexity.ai/chat/completions',
            json=payload, headers=headers
        )
        if response.status_code != 200:
            logger.error(f'Perplexity API error: {response.status_code} - {response.text}')
            raise HTTPException(status_code=502, detail=f'{response.status_code}: {response.text[:200]}')
        data = response.json()
        return data['choices'][0]['message']['content']

def strip_markdown(text: str) -> str:
    """Remove markdown formatting and citations from AI responses for plain-text display."""
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`{1,3}(.+?)`{1,3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove Perplexity citations like [1], [2][3], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*+]\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    return text.strip()

def normalize_chat_history(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Ensure chat history alternates user/assistant before sending it to Perplexity."""
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get('created_at') or '',
            0 if row.get('role') == 'user' else 1,
        ),
    )

    normalized: List[Dict[str, str]] = []
    expected_role = 'user'

    for row in sorted_rows:
        role = row.get('role')
        content = (row.get('content') or '').strip()
        if role not in {'user', 'assistant'} or not content:
            continue
        if role != expected_role:
            continue
        normalized.append({'role': role, 'content': content})
        expected_role = 'assistant' if role == 'user' else 'user'

    if normalized and normalized[-1]['role'] == 'user':
        normalized.pop()

    return normalized

def clean_json_text(text: str) -> str:
    """Clean common LLM artifacts from JSON text."""
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()

def try_repair_json(text: str) -> Optional[dict]:
    """Try to repair common JSON issues from LLMs."""
    for end_pos in range(len(text), max(0, len(text) - 20), -1):
        try:
            return json.loads(text[:end_pos])
        except Exception:
            pass
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    if open_braces > 0 or open_brackets > 0:
        for i in range(len(text) - 1, max(0, len(text) - 50), -1):
            if text[i] == ']':
                fixed = text[:i] + '}' * open_braces + text[i:]
                try:
                    return json.loads(fixed)
                except Exception:
                    pass
        fixed = text + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
        try:
            return json.loads(fixed)
        except Exception:
            pass
    if text.endswith(']}') or text.endswith('}}'):
        for trim in range(1, 5):
            try:
                return json.loads(text[:-trim])
            except Exception:
                pass
    return None

def extract_json_from_text(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(clean_json_text(match.group(1)))
            except Exception:
                pass
    start = text.find('{')
    if start != -1:
        substr = text[start:]
        try:
            return json.loads(substr)
        except Exception:
            pass
        result = try_repair_json(substr)
        if result:
            return result
    return None

# ==================== PYDANTIC MODELS ====================

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    country: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class OnboardingRequest(BaseModel):
    name: str
    age: int
    weight: float
    weight_unit: str = "kg"
    height: float
    height_unit: str = "cm"
    country: str = ""
    fitness_goal: str = "Maintain"
    activity_level: str = "Lightly Active"
    experience_level: str = "Beginner"
    workout_style: str = "Home"
    lifestyle_mode: str = "Budget-Friendly"
    my_equipment: Optional[list] = None

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    weight_unit: Optional[str] = None
    height: Optional[float] = None
    height_unit: Optional[str] = None
    country: Optional[str] = None
    fitness_goal: Optional[str] = None
    activity_level: Optional[str] = None
    experience_level: Optional[str] = None
    workout_style: Optional[str] = None
    lifestyle_mode: Optional[str] = None
    my_equipment: Optional[list] = None

class GeneratePlanRequest(BaseModel):
    days: int = 7

class MarkMealEatenRequest(BaseModel):
    plan_id: str
    day_index: int
    meal_type: str

class MarkWorkoutDoneRequest(BaseModel):
    plan_id: str
    day_index: int

class WorkoutFeedbackRequest(BaseModel):
    plan_id: str
    day_index: int
    feedback: str

class LogCustomMealRequest(BaseModel):
    meal_type: str
    name: str
    calories: int = 0

class RegenerateDayRequest(BaseModel):
    plan_id: str
    day_index: int

class ChatRequest(BaseModel):
    message: str

class LogWaterRequest(BaseModel):
    amount_ml: int

class LogStepsRequest(BaseModel):
    steps: int

class LogWeightRequest(BaseModel):
    weight: float
    unit: str = "kg"

class ReflectionRequest(BaseModel):
    mood: str
    note: str = ""

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register")
async def register(req: RegisterRequest):
    existing = supabase.table('users').select('id').eq('email', req.email.lower()).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail='Email already registered')
    user_id = str(uuid.uuid4())
    user = {
        'id': user_id, 'email': req.email.lower(),
        'password_hash': hash_password(req.password),
        'name': req.name or req.email.split('@')[0],
        'age': 25, 'weight': 70.0, 'weight_unit': 'kg',
        'height': 170.0, 'height_unit': 'cm', 'country': req.country,
        'fitness_goal': 'Maintain', 'activity_level': 'Lightly Active',
        'experience_level': 'Beginner', 'workout_style': 'Home',
        'lifestyle_mode': 'Budget-Friendly', 'onboarding_completed': False,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    supabase.table('users').insert(user).execute()
    token = create_token(user_id)
    user_resp = {k: v for k, v in user.items() if k != 'password_hash'}
    return {'token': token, 'user': user_resp}

@api_router.post("/auth/login")
async def login(req: LoginRequest):
    result = supabase.table('users').select('*').eq('email', req.email.lower()).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail='Invalid email or password')
    user = result.data[0]
    if not verify_password(req.password, user['password_hash']):
        raise HTTPException(status_code=401, detail='Invalid email or password')
    token = create_token(user['id'])
    user_resp = user_dict_from_row(user)
    return {'token': token, 'user': user_resp}

@api_router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user_dict_from_row(user)

_reset_codes: Dict[str, Dict[str, Any]] = {}

@api_router.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    email = req.email.lower().strip()
    result = supabase.table('users').select('id').eq('email', email).execute()
    if not result.data:
        return {'message': 'If that email exists, a reset code has been generated.'}
    code = ''.join(random.choices(string.digits, k=6))
    _reset_codes[email] = {'code': code, 'expires': datetime.now(timezone.utc) + timedelta(minutes=15)}
    logger.info(f'Password reset code for {email}: {code}')
    return {'message': 'If that email exists, a reset code has been generated.', 'code': code}

@api_router.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    email = req.email.lower().strip()
    stored = _reset_codes.get(email)
    if not stored or stored['code'] != req.code:
        raise HTTPException(status_code=400, detail='Invalid or expired reset code')
    if datetime.now(timezone.utc) > stored['expires']:
        _reset_codes.pop(email, None)
        raise HTTPException(status_code=400, detail='Reset code has expired')
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail='Password must be at least 6 characters')
    new_hash = hash_password(req.new_password)
    supabase.table('users').update({'password_hash': new_hash, 'updated_at': datetime.now(timezone.utc).isoformat()}).eq('email', email).execute()
    _reset_codes.pop(email, None)
    return {'message': 'Password has been reset successfully'}

# ==================== PROFILE ROUTES ====================

@api_router.post("/profile/onboarding")
async def complete_onboarding(req: OnboardingRequest, user: dict = Depends(get_current_user)):
    update_data = req.dict()
    update_data['onboarding_completed'] = True
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    supabase.table('users').update(update_data).eq('id', user['id']).execute()
    result = supabase.table('users').select('*').eq('id', user['id']).execute()
    return user_dict_from_row(result.data[0])

@api_router.put("/profile")
async def update_profile(req: ProfileUpdateRequest, user: dict = Depends(get_current_user)):
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    if update_data:
        update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
        supabase.table('users').update(update_data).eq('id', user['id']).execute()
    result = supabase.table('users').select('*').eq('id', user['id']).execute()
    return user_dict_from_row(result.data[0])

@api_router.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return user_dict_from_row(user)

# ==================== MEAL PLAN ROUTES ====================

def build_fallback_meal_plan(days: int, lifestyle: str):
    tag = "Budget pick" if lifestyle == "Budget-Friendly" else "Full access"
    meals_pool = [
        {"breakfast": {"name": "Oatmeal with Banana", "calories": 320, "description": "Creamy oats topped with sliced banana and honey", "ingredients": ["oats", "banana", "honey", "milk"], "preparation": "Cook oats with milk, top with banana slices and drizzle honey", "tag": tag},
         "lunch": {"name": "Chicken & Rice Bowl", "calories": 480, "description": "Grilled chicken breast with steamed rice and veggies", "ingredients": ["chicken breast", "rice", "broccoli", "soy sauce"], "preparation": "Grill chicken, steam rice and broccoli, season with soy sauce", "tag": tag},
         "dinner": {"name": "Pasta with Tomato Sauce", "calories": 520, "description": "Whole wheat pasta with homemade tomato sauce", "ingredients": ["whole wheat pasta", "tomatoes", "garlic", "olive oil", "basil"], "preparation": "Cook pasta, make sauce with tomatoes garlic and basil", "tag": tag},
         "snack": {"name": "Greek Yogurt & Nuts", "calories": 180, "description": "Protein-rich yogurt with mixed nuts", "ingredients": ["greek yogurt", "almonds", "walnuts"], "preparation": "Top yogurt with crushed nuts", "tag": tag}},
        {"breakfast": {"name": "Scrambled Eggs on Toast", "calories": 380, "description": "Fluffy scrambled eggs on whole grain toast", "ingredients": ["eggs", "bread", "butter", "salt", "pepper"], "preparation": "Scramble eggs in butter, serve on toasted bread", "tag": tag},
         "lunch": {"name": "Tuna Salad Wrap", "calories": 420, "description": "Tuna mixed with veggies in a tortilla wrap", "ingredients": ["canned tuna", "tortilla", "lettuce", "tomato", "mayo"], "preparation": "Mix tuna with mayo, add veggies, wrap in tortilla", "tag": tag},
         "dinner": {"name": "Stir-Fry Vegetables with Tofu", "calories": 450, "description": "Colorful stir-fry with crispy tofu", "ingredients": ["tofu", "bell peppers", "carrots", "soy sauce", "rice"], "preparation": "Fry tofu until crispy, stir-fry vegetables, serve over rice", "tag": tag},
         "snack": {"name": "Apple with Peanut Butter", "calories": 200, "description": "Sliced apple with peanut butter dip", "ingredients": ["apple", "peanut butter"], "preparation": "Slice apple and serve with peanut butter", "tag": tag}},
    ]
    result = []
    for i in range(days):
        meals = meals_pool[i % len(meals_pool)]
        result.append({"day": i + 1, "meals": meals})
    return result

@api_router.post("/meals/generate")
async def generate_meal_plan(req: GeneratePlanRequest, user: dict = Depends(get_current_user)):
    lifestyle = user.get('lifestyle_mode', 'Budget-Friendly')
    country = user.get('country') or 'a neutral location'
    
    budget_instruction = "IMPORTANT: Only use highly affordable, culturally local, and commonly available ingredients found in wet markets/grocery stores." if lifestyle == "Budget-Friendly" else "Allow varied/premium ingredients."
    tag = "Budget pick" if lifestyle == "Budget-Friendly" else "Full access"

    system_prompt = f"You are an expert nutritionist native to {country}. You MUST respond with ONLY a raw JSON object. NO markdown, NO code blocks (```), NO introductory text, NO conversational replies. Output ONLY valid parseable JSON."
    user_prompt = f"""Plan a healthy {req.days}-day meal plan explicitly incorporating {country} cuisine and local staples. User: age {user.get('age', 25)}, {user.get('weight', 70)}{user.get('weight_unit', 'kg')}, goal: {user.get('fitness_goal', 'Maintain')}, activity: {user.get('activity_level', 'Lightly Active')}. Context: Location is {country}. Lifestyle is {lifestyle}. {budget_instruction}

Output EXACTLY this JSON format (replace ... with data). Strict compliance required:
{{"days":[{{"day":1,"meals":{{"breakfast":{{"name":"...","calories":350,"description":"...","ingredients":["..."],"preparation":"...","tag":"{tag}"}},"lunch":{{...}},"dinner":{{...}},"snack":{{...}}}}}}]}}"""

    try:
        response_text = await call_perplexity(system_prompt, user_prompt, max_tokens=4000)
        logger.info(f'Perplexity meal response (first 500 chars): {response_text[:500]}')
        plan_data = extract_json_from_text(response_text)
        logger.info(f'Extracted JSON keys: {list(plan_data.keys()) if plan_data else "NONE"}')
        if not plan_data or 'days' not in plan_data:
            logger.warning('Failed to extract meal plan JSON, using fallback')
            plan_data = {"days": build_fallback_meal_plan(req.days, lifestyle)}
    except Exception as e:
        logger.warning(f'Perplexity call failed, using fallback: {e}')
        plan_data = {"days": build_fallback_meal_plan(req.days, lifestyle)}

    plan_id = str(uuid.uuid4())
    plan_doc = {
        'id': plan_id, 'user_id': user['id'], 'days': req.days,
        'plan_data': plan_data['days'], 'eaten_meals': [],
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    supabase.table('meal_plans').insert(plan_doc).execute()
    await check_achievements(user['id'])
    return plan_doc

@api_router.get("/meals/current")
async def get_current_meal_plan(user: dict = Depends(get_current_user)):
    result = supabase.table('meal_plans').select('*').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
    return result.data[0] if result.data else None

@api_router.post("/meals/eaten")
async def mark_meal_eaten(req: MarkMealEatenRequest, user: dict = Depends(get_current_user)):
    meal_entry = {'day_index': req.day_index, 'meal_type': req.meal_type}
    # Get current eaten_meals
    result = supabase.table('meal_plans').select('eaten_meals').eq('id', req.plan_id).eq('user_id', user['id']).execute()
    if result.data:
        eaten = result.data[0].get('eaten_meals', [])
        if meal_entry not in eaten:
            eaten.append(meal_entry)
        supabase.table('meal_plans').update({'eaten_meals': eaten}).eq('id', req.plan_id).execute()

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    # Upsert daily log
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if log_result.data:
        meals_eaten = log_result.data[0].get('meals_eaten', [])
        if meal_entry not in meals_eaten:
            meals_eaten.append(meal_entry)
        supabase.table('daily_logs').update({
            'meals_eaten': meals_eaten,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('user_id', user['id']).eq('date', today).execute()
    else:
        supabase.table('daily_logs').insert({
            'user_id': user['id'], 'date': today,
            'meals_eaten': [meal_entry],
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()

    await update_streak(user['id'])
    return {'status': 'ok'}

@api_router.post("/meals/log-custom")
async def log_custom_meal(req: LogCustomMealRequest, user: dict = Depends(get_current_user)):
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = {'meal_type': req.meal_type, 'name': req.name, 'calories': req.calories, 'custom': True}
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if log_result.data:
        custom_meals = log_result.data[0].get('custom_meals', []) or []
        custom_meals.append(entry)
        supabase.table('daily_logs').update({
            'custom_meals': custom_meals,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('user_id', user['id']).eq('date', today).execute()
    else:
        supabase.table('daily_logs').insert({
            'user_id': user['id'], 'date': today,
            'custom_meals': [entry],
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()
    await update_streak(user['id'])
    return {'status': 'ok', 'entry': entry}

@api_router.get("/meals/custom-today")
async def get_custom_meals_today(user: dict = Depends(get_current_user)):
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    result = supabase.table('daily_logs').select('custom_meals').eq('user_id', user['id']).eq('date', today).execute()
    if result.data and result.data[0].get('custom_meals'):
        return result.data[0]['custom_meals']
    return []

@api_router.post("/meals/regenerate-day")
async def regenerate_meal_day(req: RegenerateDayRequest, user: dict = Depends(get_current_user)):
    result = supabase.table('meal_plans').select('*').eq('id', req.plan_id).eq('user_id', user['id']).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail='Plan not found')
    plan = result.data[0]
    lifestyle = user.get('lifestyle_mode', 'Budget-Friendly')
    tag = "Budget pick" if lifestyle == "Budget-Friendly" else "Full access"
    system_prompt = "You are a nutritionist. Return ONLY valid JSON."
    user_prompt = f"""Generate meals for 1 day. Goal: {user.get('fitness_goal', 'Maintain')}, Activity: {user.get('activity_level', 'Lightly Active')}, Lifestyle: {lifestyle}.
Return ONLY JSON: {{"day":{req.day_index + 1},"meals":{{"breakfast":{{"name":"...","calories":350,"description":"...","ingredients":["..."],"preparation":"...","tag":"{tag}"}},"lunch":{{...}},"dinner":{{...}},"snack":{{...}}}}}}"""
    try:
        response_text = await call_perplexity(system_prompt, user_prompt, max_tokens=1500)
        day_data = extract_json_from_text(response_text)
        if day_data and 'meals' in day_data:
            plan_data = plan['plan_data']
            plan_data[req.day_index] = day_data
            supabase.table('meal_plans').update({'plan_data': plan_data}).eq('id', req.plan_id).execute()
            return {'status': 'ok', 'day': day_data}
    except Exception as e:
        logger.error(f'Regenerate meal day error: {e}')
    raise HTTPException(status_code=500, detail='Failed to regenerate day')

# ==================== WORKOUT PLAN ROUTES ====================

def build_fallback_workout_plan(days: int, lifestyle: str):
    exercises_pool = [
        {"day": 1, "workout_type": "Full Body", "duration": "30 mins", "intensity": "Moderate",
         "exercises": [
             {"name": "Push-ups", "sets": 3, "reps": "12", "description": "Standard push-ups on the floor", "equipment": "None", "rest": "60s"},
             {"name": "Bodyweight Squats", "sets": 3, "reps": "15", "description": "Standard squats", "equipment": "None", "rest": "60s"},
             {"name": "Plank", "sets": 3, "reps": "30s", "description": "Hold plank position", "equipment": "None", "rest": "45s"},
             {"name": "Lunges", "sets": 3, "reps": "10 each", "description": "Alternating forward lunges", "equipment": "None", "rest": "60s"},
         ]},
        {"day": 2, "workout_type": "Upper Body", "duration": "25 mins", "intensity": "Moderate",
         "exercises": [
             {"name": "Diamond Push-ups", "sets": 3, "reps": "10", "description": "Hands close together push-ups", "equipment": "None", "rest": "60s"},
             {"name": "Tricep Dips", "sets": 3, "reps": "12", "description": "Using a chair or bench", "equipment": "Chair", "rest": "60s"},
             {"name": "Pike Push-ups", "sets": 3, "reps": "8", "description": "Elevated pike position push-ups", "equipment": "None", "rest": "60s"},
             {"name": "Arm Circles", "sets": 3, "reps": "30s each direction", "description": "Large arm circles", "equipment": "None", "rest": "30s"},
         ]},
        {"day": 3, "workout_type": "Lower Body", "duration": "30 mins", "intensity": "Hard",
         "exercises": [
             {"name": "Jump Squats", "sets": 3, "reps": "12", "description": "Explosive squat jumps", "equipment": "None", "rest": "60s"},
             {"name": "Wall Sit", "sets": 3, "reps": "45s", "description": "Hold wall sit position", "equipment": "Wall", "rest": "60s"},
             {"name": "Calf Raises", "sets": 3, "reps": "20", "description": "Standing calf raises", "equipment": "None", "rest": "45s"},
             {"name": "Glute Bridges", "sets": 3, "reps": "15", "description": "Lying glute bridges", "equipment": "None", "rest": "60s"},
         ]},
        {"day": 4, "workout_type": "Core & Cardio", "duration": "25 mins", "intensity": "Moderate",
         "exercises": [
             {"name": "Mountain Climbers", "sets": 3, "reps": "30s", "description": "Fast mountain climbers", "equipment": "None", "rest": "45s"},
             {"name": "Bicycle Crunches", "sets": 3, "reps": "20", "description": "Alternating bicycle crunches", "equipment": "None", "rest": "45s"},
             {"name": "Burpees", "sets": 3, "reps": "8", "description": "Full burpees with jump", "equipment": "None", "rest": "60s"},
             {"name": "Russian Twists", "sets": 3, "reps": "20", "description": "Seated torso twists", "equipment": "None", "rest": "45s"},
         ]},
    ]
    result = []
    for i in range(days):
        day = dict(exercises_pool[i % len(exercises_pool)])
        day['day'] = i + 1
        result.append(day)
    return result

@api_router.post("/workouts/generate")
async def generate_workout_plan(req: GeneratePlanRequest, user: dict = Depends(get_current_user)):
    lifestyle = user.get('lifestyle_mode', 'Budget-Friendly')
    workout_style = user.get('workout_style', 'Home')
    country = user.get('country') or 'a neutral location'

    my_equipment = user.get('my_equipment') or []
    if my_equipment:
        equipment_note = f"The user also owns this equipment: {', '.join(my_equipment)}. Where appropriate, incorporate these for more effective exercises. Bodyweight exercises are always fine too."
    elif lifestyle == "Budget-Friendly":
        equipment_note = "NO equipment, strictly bodyweight only. Home-based exercises ONLY."
    elif workout_style == "Gym":
        equipment_note = "Gym equipment: dumbbells, barbells, machines, cables."
    elif workout_style == "Outdoor":
        equipment_note = "Outdoor exercises: running, park workouts, bodyweight."
    else:
        equipment_note = "Home with basic equipment: dumbbells, resistance bands, yoga mat."

    system_prompt = f"You are an expert {country} fitness coach. You MUST respond with ONLY a raw JSON object. NO markdown, NO code blocks (```), NO introductory text, NO conversational replies. Output ONLY valid parseable JSON."
    user_prompt = f"""Create a {req.days}-day workout plan. User goal: {user.get('fitness_goal', 'Maintain')}, level: {user.get('experience_level', 'Beginner')}, activity: {user.get('activity_level', 'Lightly Active')}, style: {workout_style}. Context: {country}. {equipment_note}

Output EXACTLY this JSON format (replace ... with data, 4-5 exercises per day). Strict compliance required:
{{"days":[{{"day":1,"workout_type":"...","duration":"30 mins","intensity":"Moderate","exercises":[{{"name":"...","sets":3,"reps":"12","description":"...","equipment":"None","rest":"60s"}}]}}]}}"""

    try:
        response_text = await call_perplexity(system_prompt, user_prompt, max_tokens=4000)
        plan_data = extract_json_from_text(response_text)
        if not plan_data or 'days' not in plan_data:
            plan_data = {"days": build_fallback_workout_plan(req.days, lifestyle)}
    except Exception as e:
        logger.warning(f'Perplexity call failed, using fallback: {e}')
        plan_data = {"days": build_fallback_workout_plan(req.days, lifestyle)}

    plan_id = str(uuid.uuid4())
    plan_doc = {
        'id': plan_id, 'user_id': user['id'], 'days': req.days,
        'plan_data': plan_data['days'], 'completed_days': [], 'feedback': [],
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    supabase.table('workout_plans').insert(plan_doc).execute()
    return plan_doc

@api_router.get("/workouts/current")
async def get_current_workout_plan(user: dict = Depends(get_current_user)):
    result = supabase.table('workout_plans').select('*').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
    return result.data[0] if result.data else None

@api_router.post("/workouts/complete")
async def mark_workout_done(req: MarkWorkoutDoneRequest, user: dict = Depends(get_current_user)):
    # Update completed_days in workout plan
    result = supabase.table('workout_plans').select('completed_days').eq('id', req.plan_id).eq('user_id', user['id']).execute()
    if result.data:
        completed = result.data[0].get('completed_days', [])
        if req.day_index not in completed:
            completed.append(req.day_index)
        supabase.table('workout_plans').update({'completed_days': completed}).eq('id', req.plan_id).execute()

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if log_result.data:
        workouts = log_result.data[0].get('workouts_completed', [])
        if req.day_index not in workouts:
            workouts.append(req.day_index)
        supabase.table('daily_logs').update({
            'workouts_completed': workouts,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('user_id', user['id']).eq('date', today).execute()
    else:
        supabase.table('daily_logs').insert({
            'user_id': user['id'], 'date': today,
            'workouts_completed': [req.day_index],
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()

    await update_streak(user['id'])
    await check_achievements(user['id'])
    return {'status': 'ok'}

@api_router.post("/workouts/feedback")
async def submit_workout_feedback(req: WorkoutFeedbackRequest, user: dict = Depends(get_current_user)):
    result = supabase.table('workout_plans').select('feedback').eq('id', req.plan_id).eq('user_id', user['id']).execute()
    if result.data:
        feedback_list = result.data[0].get('feedback', [])
        feedback_entry = {'day_index': req.day_index, 'feedback': req.feedback, 'created_at': datetime.now(timezone.utc).isoformat()}
        feedback_list.append(feedback_entry)
        supabase.table('workout_plans').update({'feedback': feedback_list}).eq('id', req.plan_id).execute()
    return {'status': 'ok'}

@api_router.post("/workouts/regenerate-day")
async def regenerate_workout_day(req: RegenerateDayRequest, user: dict = Depends(get_current_user)):
    result = supabase.table('workout_plans').select('*').eq('id', req.plan_id).eq('user_id', user['id']).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail='Plan not found')
    plan = result.data[0]
    lifestyle = user.get('lifestyle_mode', 'Budget-Friendly')
    equip = "bodyweight only, no equipment" if lifestyle == "Budget-Friendly" else "basic equipment available"
    system_prompt = "You are a fitness coach. Return ONLY valid JSON."
    user_prompt = f"""Generate 1 workout day. Goal: {user.get('fitness_goal', 'Maintain')}, Level: {user.get('experience_level', 'Beginner')}, Equipment: {equip}.
Return ONLY JSON: {{"day":{req.day_index + 1},"workout_type":"...","duration":"30 mins","intensity":"Moderate","exercises":[{{"name":"...","sets":3,"reps":"12","description":"...","equipment":"None","rest":"60s"}}]}}"""
    try:
        response_text = await call_perplexity(system_prompt, user_prompt, max_tokens=1500)
        day_data = extract_json_from_text(response_text)
        if day_data and 'exercises' in day_data:
            plan_data = plan['plan_data']
            plan_data[req.day_index] = day_data
            supabase.table('workout_plans').update({'plan_data': plan_data}).eq('id', req.plan_id).execute()
            return {'status': 'ok', 'day': day_data}
    except Exception as e:
        logger.error(f'Regenerate workout day error: {e}')
    raise HTTPException(status_code=500, detail='Failed to regenerate day')

# ==================== AI COACH ROUTES ====================

@api_router.post("/ai/chat")
async def ai_chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    # Get recent daily logs for context
    week_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
    recent_logs_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).gte('date', week_start).order('date', desc=True).limit(7).execute()
    recent_logs = recent_logs_result.data if recent_logs_result.data else []
    workouts_this_week = sum(1 for log in recent_logs if log.get('workouts_completed'))

    # Fetch User's current plans
    meal_plan_result = supabase.table('meal_plans').select('plan_data,eaten_meals').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
    meal_plan_ctx = "No meal plan generated yet."
    if meal_plan_result.data:
        mp = meal_plan_result.data[0]
        eaten_count = len(mp.get('eaten_meals', []))
        meal_plan_ctx = f"Has active meal plan. Details: {json.dumps(mp.get('plan_data', [])[:2])} (truncated). Total meals marked eaten: {eaten_count}."

    workout_plan_result = supabase.table('workout_plans').select('plan_data,completed_days').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
    workout_plan_ctx = "No workout plan generated yet."
    if workout_plan_result.data:
        wp = workout_plan_result.data[0]
        completed_days = wp.get('completed_days', [])
        workout_plan_ctx = f"Has active workout plan. Details: {json.dumps(wp.get('plan_data', [])[:2])} (truncated). Completed days: {completed_days}."

    # Fetch recent chat history
    history_result = supabase.table('ai_messages').select('*').eq('user_id', user['id']).order('created_at', desc=True).limit(14).execute()
    history_rows = history_result.data if history_result.data else []
    past_messages = normalize_chat_history(list(reversed(history_rows)))

    lifestyle = user.get('lifestyle_mode', 'Budget-Friendly')
    lifestyle_desc = ("Budget-Friendly (they are on a strict budget/less money, suggest cheap foods and free/no-equipment workouts)" 
                      if lifestyle == "Budget-Friendly" else "Full Access (they have resources/money, suggest premium healthy foods and gym access/gear)")

    system_prompt = f"""You are Era, a friendly AI fitness coach in the MovEra app.

User Profile:
- Name: {user.get('name') or 'User'}, Age: {user.get('age', 25)}
- Weight: {user.get('weight', 70)} {user.get('weight_unit', 'kg')}, Height: {user.get('height', 170)} {user.get('height_unit', 'cm')}
- Goal: {user.get('fitness_goal', 'Maintain')}, Activity: {user.get('activity_level', 'Lightly Active')}
- Experience: {user.get('experience_level', 'Beginner')}, Style: {user.get('workout_style', 'Home')}
- Lifestyle Mode: {lifestyle_desc}
- Workouts this week: {workouts_this_week}

User Plans Context (use this strictly if they ask about their current diet, meals, or workouts):
- Meal Plan Status: {meal_plan_ctx}
- Workout Plan Status: {workout_plan_ctx}

Guidelines:
- Be encouraging, practical, and incredibly concise (under 200 words).
- Respect their Lifestyle Mode strictly in advice.
- You MUST remember past conversations -- they are provided to you right before the users latest message. Reply contextually.
- Never diagnose medical conditions."""

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(past_messages)
    messages.append({'role': 'user', 'content': req.message})

    try:
        response_text = await call_perplexity_chat(messages, max_tokens=1500)
        response_text = strip_markdown(response_text)
        user_created_at = datetime.now(timezone.utc)
        assistant_created_at = user_created_at + timedelta(milliseconds=1)
        supabase.table('ai_messages').insert([
            {'id': str(uuid.uuid4()), 'user_id': user['id'], 'role': 'user', 'content': req.message, 'created_at': user_created_at.isoformat()},
            {'id': str(uuid.uuid4()), 'user_id': user['id'], 'role': 'assistant', 'content': response_text, 'created_at': assistant_created_at.isoformat()}
        ]).execute()
        await check_achievements(user['id'])
        return {'message': response_text}
    except Exception as e:
        logger.error(f'AI chat error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/ai/history")
async def get_chat_history(user: dict = Depends(get_current_user)):
    result = supabase.table('ai_messages').select('id,user_id,role,content,created_at').eq('user_id', user['id']).order('created_at').limit(100).execute()
    return result.data if result.data else []

@api_router.delete("/ai/history")
async def clear_chat_history(user: dict = Depends(get_current_user)):
    supabase.table('ai_messages').delete().eq('user_id', user['id']).execute()
    return {'status': 'ok'}

# ==================== DAILY LOG ROUTES ====================

def get_today():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')

@api_router.post("/logs/water")
async def log_water(req: LogWaterRequest, user: dict = Depends(get_current_user)):
    today = get_today()
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if log_result.data:
        current_water = log_result.data[0].get('water_ml', 0)
        new_water = max(0, current_water + req.amount_ml)
        supabase.table('daily_logs').update({
            'water_ml': new_water,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('user_id', user['id']).eq('date', today).execute()
    else:
        supabase.table('daily_logs').insert({
            'user_id': user['id'], 'date': today,
            'water_ml': req.amount_ml,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()
    log = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    row = log.data[0]
    await check_achievements(user['id'])
    return {k: v for k, v in row.items() if k != 'id'}

@api_router.post("/logs/steps")
async def log_steps(req: LogStepsRequest, user: dict = Depends(get_current_user)):
    today = get_today()
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if log_result.data:
        current_steps = log_result.data[0].get('steps', 0)
        new_steps = max(current_steps, req.steps)
        supabase.table('daily_logs').update({
            'steps': new_steps,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).eq('user_id', user['id']).eq('date', today).execute()
    else:
        supabase.table('daily_logs').insert({
            'user_id': user['id'], 'date': today,
            'steps': req.steps,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()
    log = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    row = log.data[0]
    await check_achievements(user['id'])
    return {k: v for k, v in row.items() if k != 'id'}

@api_router.post("/logs/weight")
async def log_weight(req: LogWeightRequest, user: dict = Depends(get_current_user)):
    today = get_today()
    supabase.table('weight_logs').insert({
        'weight': req.weight, 'unit': req.unit, 'date': today,
        'user_id': user['id'], 'created_at': datetime.now(timezone.utc).isoformat()
    }).execute()
    supabase.table('users').update({'weight': req.weight, 'weight_unit': req.unit}).eq('id', user['id']).execute()
    return {'status': 'ok'}

@api_router.get("/logs/daily")
async def get_daily_log(user: dict = Depends(get_current_user)):
    today = get_today()
    result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    if result.data:
        row = result.data[0]
        return {k: v for k, v in row.items() if k != 'id'}
    return {'user_id': user['id'], 'date': today, 'water_ml': 0, 'steps': 0, 'meals_eaten': [], 'workouts_completed': []}

@api_router.get("/logs/weight-history")
async def get_weight_history(user: dict = Depends(get_current_user)):
    result = supabase.table('weight_logs').select('weight,unit,date,created_at').eq('user_id', user['id']).order('date', desc=True).limit(30).execute()
    entries = list(reversed(result.data)) if result.data else []
    return entries

@api_router.get("/logs/weekly-summary")
async def get_weekly_summary(user: dict = Depends(get_current_user)):
    today = datetime.now(timezone.utc).date()
    days = []
    total_water = 0
    total_steps = 0
    total_workouts = 0
    total_meals = 0
    for i in range(6, -1, -1):
        date_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', date_str).execute()
        log = result.data[0] if result.data else {}
        water = log.get('water_ml', 0) or 0
        steps = log.get('steps', 0) or 0
        workouts = len(log.get('workouts_completed', []) or [])
        meals = len(log.get('meals_eaten', []) or [])
        total_water += water
        total_steps += steps
        total_workouts += workouts
        total_meals += meals
        days.append({'date': date_str, 'water_ml': water, 'steps': steps, 'workouts': workouts, 'meals': meals, 'active': bool(workouts or meals)})
    return {
        'days': days,
        'avg_water': round(total_water / 7),
        'avg_steps': round(total_steps / 7),
        'total_workouts': total_workouts,
        'total_meals': total_meals,
    }

# ==================== DASHBOARD ====================

MOTIVATIONAL_QUOTES = [
    "The only bad workout is the one that didn't happen.",
    "Your body can stand almost anything. It's your mind you have to convince.",
    "Success is the sum of small efforts repeated day in and day out.",
    "Don't stop when you're tired. Stop when you're done.",
    "The pain you feel today will be the strength you feel tomorrow.",
    "Push harder than yesterday if you want a different tomorrow.",
    "Your health is an investment, not an expense.",
    "Fitness is not about being better than someone else. It's about being better than you used to be.",
    "The secret of getting ahead is getting started.",
    "Take care of your body. It's the only place you have to live.",
    "A one-hour workout is 4% of your day. No excuses.",
    "Strive for progress, not perfection.",
]

@api_router.get("/dashboard")
async def get_dashboard(user: dict = Depends(get_current_user)):
    today = get_today()
    log_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).eq('date', today).execute()
    daily_log = log_result.data[0] if log_result.data else {'water_ml': 0, 'steps': 0, 'meals_eaten': [], 'workouts_completed': []}

    meal_result = supabase.table('meal_plans').select('id').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
    workout_result = supabase.table('workout_plans').select('id').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()

    workout_status = 'Not started'
    if daily_log.get('workouts_completed'):
        workout_status = 'Completed'

    week_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
    weekly_result = supabase.table('daily_logs').select('*').eq('user_id', user['id']).gte('date', week_start).execute()
    weekly_logs = weekly_result.data if weekly_result.data else []

    workouts_this_week = sum(1 for log in weekly_logs if log.get('workouts_completed'))
    meal_days_followed = sum(1 for log in weekly_logs if log.get('meals_eaten'))
    streak = await get_streak(user['id'])

    total_calories = 0
    custom_meals = daily_log.get('custom_meals', []) or []
    custom_meal_types = {cm.get('meal_type') for cm in custom_meals}

    for cm in custom_meals:
        total_calories += cm.get('calories', 0) or 0

    eaten_meals = daily_log.get('meals_eaten', []) or []
    ai_cal_count = 0
    if eaten_meals:
        current_plan = supabase.table('meal_plans').select('plan_data').eq('user_id', user['id']).order('created_at', desc=True).limit(1).execute()
        if current_plan.data and current_plan.data[0].get('plan_data'):
            plan_days = current_plan.data[0]['plan_data']
            for eaten in eaten_meals:
                di = eaten.get('day_index', 0)
                mt = eaten.get('meal_type', '')
                if mt in custom_meal_types:
                    continue
                if di < len(plan_days):
                    meal = plan_days[di].get('meals', {}).get(mt, {})
                    ai_cal_count += meal.get('calories', 0) or 0
        if ai_cal_count == 0 and not custom_meal_types:
            ai_cal_count = len(eaten_meals) * 400
    total_calories += ai_cal_count

    calorie_goals = {'Lose Weight': 1800, 'Maintain': 2200, 'Build Muscle': 2800}
    calorie_goal = calorie_goals.get(user.get('fitness_goal', 'Maintain'), 2200)

    day_hash = int(hashlib.md5(today.encode()).hexdigest(), 16)
    quote = MOTIVATIONAL_QUOTES[day_hash % len(MOTIVATIONAL_QUOTES)]

    return {
        'greeting_name': user.get('name') or 'User',
        'calories_goal': calorie_goal,
        'calories_consumed': total_calories,
        'water_ml': daily_log.get('water_ml', 0),
        'steps': daily_log.get('steps', 0),
        'workout_status': workout_status,
        'quote': quote,
        'workouts_this_week': workouts_this_week,
        'workout_target': 5,
        'meal_days_followed': meal_days_followed,
        'streak': streak,
        'has_meal_plan': bool(meal_result.data),
        'has_workout_plan': bool(workout_result.data),
    }

# ==================== STREAKS & ACHIEVEMENTS ====================

async def get_streak(user_id: str) -> int:
    today = datetime.now(timezone.utc).date()
    streak = 0
    for i in range(365):
        date_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        result = supabase.table('daily_logs').select('meals_eaten,workouts_completed').eq('user_id', user_id).eq('date', date_str).execute()
        if result.data:
            log = result.data[0]
            if log.get('workouts_completed') or log.get('meals_eaten'):
                streak += 1
            elif i > 0:
                break
        elif i > 0:
            break
    return streak

async def update_streak(user_id: str):
    streak = await get_streak(user_id)
    supabase.table('users').update({'current_streak': streak}).eq('id', user_id).execute()

async def check_achievements(user_id: str):
    streak = await get_streak(user_id)
    workout_logs = supabase.table('daily_logs').select('id').eq('user_id', user_id).neq('workouts_completed', '[]').execute()
    total_workouts = len(workout_logs.data) if workout_logs.data else 0
    user_result = supabase.table('users').select('lifestyle_mode').eq('id', user_id).execute()
    user = user_result.data[0] if user_result.data else {}

    today = get_today()
    today_log = supabase.table('daily_logs').select('water_ml,steps').eq('user_id', user_id).eq('date', today).execute()
    today_data = today_log.data[0] if today_log.data else {}
    today_water = today_data.get('water_ml', 0) or 0
    today_steps = today_data.get('steps', 0) or 0

    meal_plans_result = supabase.table('meal_plans').select('id').eq('user_id', user_id).execute()
    total_meal_plans = len(meal_plans_result.data) if meal_plans_result.data else 0

    chat_count_result = supabase.table('ai_messages').select('id').eq('user_id', user_id).eq('role', 'user').execute()
    total_chats = len(chat_count_result.data) if chat_count_result.data else 0

    badges = []
    if total_workouts >= 1:
        badges.append({'badge_id': 'first_workout', 'name': 'First Workout', 'description': 'Completed your first workout!', 'icon': 'trophy'})
    if streak >= 3:
        badges.append({'badge_id': '3_day_streak', 'name': '3-Day Streak', 'description': '3 consecutive active days!', 'icon': 'flame'})
    if streak >= 7:
        badges.append({'badge_id': '7_day_streak', 'name': '7-Day Streak', 'description': 'A full week of consistency!', 'icon': 'star'})
    if streak >= 14:
        badges.append({'badge_id': '14_day_streak', 'name': 'Consistent', 'description': '14 days of pure dedication!', 'icon': 'ribbon'})
    if streak >= 30:
        badges.append({'badge_id': '30_day_streak', 'name': 'Iron Will', 'description': '30 days strong. Unstoppable!', 'icon': 'shield'})
    if user.get('lifestyle_mode') == 'Budget-Friendly' and total_workouts >= 7:
        badges.append({'badge_id': 'budget_beast', 'name': 'Budget Beast', 'description': 'Crushing it on Budget-Friendly!', 'icon': 'flash'})
    if total_meal_plans >= 1:
        badges.append({'badge_id': 'meal_prepper', 'name': 'Meal Prepper', 'description': 'Generated your first meal plan!', 'icon': 'restaurant'})
    if today_water >= 2000:
        badges.append({'badge_id': 'hydration_hero', 'name': 'Hydration Hero', 'description': 'Drank 2L+ water in a day!', 'icon': 'water'})
    if today_steps >= 10000:
        badges.append({'badge_id': 'step_master', 'name': 'Step Master', 'description': 'Hit 10,000 steps in a day!', 'icon': 'footsteps'})
    if total_chats >= 50:
        badges.append({'badge_id': 'chatterbox', 'name': 'Chatterbox', 'description': 'Sent 50+ messages to Era!', 'icon': 'chatbubbles'})
    if total_workouts >= 7:
        badges.append({'badge_id': 'full_week', 'name': 'Full Week', 'description': '7 workout days completed!', 'icon': 'calendar'})
    if total_workouts >= 100:
        badges.append({'badge_id': 'centurion', 'name': 'Centurion', 'description': '100 workout days. Legend!', 'icon': 'medal'})

    for badge in badges:
        existing = supabase.table('achievements').select('id').eq('user_id', user_id).eq('badge_id', badge['badge_id']).execute()
        if not existing.data:
            supabase.table('achievements').insert({
                'id': str(uuid.uuid4()), 'user_id': user_id,
                'badge_id': badge['badge_id'], 'name': badge['name'],
                'description': badge['description'], 'icon': badge['icon'],
                'earned_at': datetime.now(timezone.utc).isoformat()
            }).execute()

@api_router.get("/achievements")
async def get_achievements(user: dict = Depends(get_current_user)):
    result = supabase.table('achievements').select('id,user_id,badge_id,name,description,icon,earned_at').eq('user_id', user['id']).execute()
    return result.data if result.data else []

@api_router.get("/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    workout_logs = supabase.table('daily_logs').select('id').eq('user_id', user['id']).neq('workouts_completed', '[]').execute()
    total_workouts = len(workout_logs.data) if workout_logs.data else 0
    meal_plans_result = supabase.table('meal_plans').select('id').eq('user_id', user['id']).execute()
    total_meal_plans = len(meal_plans_result.data) if meal_plans_result.data else 0
    streak = await get_streak(user['id'])
    achievements_result = supabase.table('achievements').select('id,user_id,badge_id,name,description,icon,earned_at').eq('user_id', user['id']).execute()
    achievements = achievements_result.data if achievements_result.data else []
    return {'total_workouts': total_workouts, 'total_meal_plans': total_meal_plans, 'current_streak': streak, 'achievements': achievements}

# ==================== REFLECTIONS ====================

@api_router.post("/reflections")
async def submit_reflection(req: ReflectionRequest, user: dict = Depends(get_current_user)):
    today = get_today()
    reflection = {
        'id': str(uuid.uuid4()), 'user_id': user['id'], 'date': today,
        'mood': req.mood, 'note': req.note,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    # Upsert: delete existing for today, then insert
    supabase.table('daily_reflections').delete().eq('user_id', user['id']).eq('date', today).execute()
    supabase.table('daily_reflections').insert(reflection).execute()
    messages = {
        'sad': "It's okay to have tough days. Tomorrow is a fresh start!",
        'neutral': "Every step counts. Keep going!",
        'happy': "Amazing energy! Keep riding that momentum!"
    }
    return {'message': messages.get(req.mood, 'Keep going!'), 'reflection': reflection}

@api_router.get("/reflections/today")
async def get_today_reflection(user: dict = Depends(get_current_user)):
    today = get_today()
    result = supabase.table('daily_reflections').select('*').eq('user_id', user['id']).eq('date', today).execute()
    return result.data[0] if result.data else None

@api_router.get("/reflections/history")
async def get_reflection_history(user: dict = Depends(get_current_user)):
    result = supabase.table('daily_reflections').select('*').eq('user_id', user['id']).order('date', desc=True).limit(30).execute()
    return result.data if result.data else []

# ==================== ACCOUNT ====================

@api_router.delete("/account")
async def delete_account(user: dict = Depends(get_current_user)):
    uid = user['id']
    # Delete from all related tables (cascading should handle most, but be explicit)
    for table in ['ai_messages', 'achievements', 'daily_reflections', 'daily_logs', 'weight_logs', 'meal_plans', 'workout_plans']:
        supabase.table(table).delete().eq('user_id', uid).execute()
    supabase.table('users').delete().eq('id', uid).execute()
    return {'status': 'deleted'}

# ==================== APP SETUP ====================

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok", "app": "MovEra API", "docs": "/docs"}

@app.get("/debug/perplexity")
async def debug_perplexity():
    """Temporary endpoint to diagnose Perplexity API issues on Render."""
    key = PERPLEXITY_API_KEY
    key_info = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else f"len={len(key)}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                'https://api.perplexity.ai/chat/completions',
                json={'model': PERPLEXITY_MODEL, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 10},
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
            )
            return {"key_preview": key_info, "status": response.status_code, "body": response.text[:300]}
    except Exception as e:
        return {"key_preview": key_info, "error": str(e)}

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
