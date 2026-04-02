import os
import bcrypt
from supabase import create_client, Client


def get_admin_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    # Fail fast with a clear message instead of a cryptic supabase error
    if not url:
        raise EnvironmentError(
            "SUPABASE_URL environment variable is not set. "
            "Add it to your Render environment variables."
        )
    if not key:
        raise EnvironmentError(
            "SUPABASE_SERVICE_KEY environment variable is not set. "
            "Add it to your Render environment variables."
        )

    return create_client(url, key)


# ── Auth operations ───────────────────────────────────────────
def register_user(email, password):
    """Register user directly in database with hashed password"""
    try:
        admin = get_admin_client()

        # Check if email already exists
        existing = admin.table("users") \
                        .select("id") \
                        .eq("email", email) \
                        .execute()

        if existing.data:
            return {"success": False, "error": "Email already registered"}

        # Hash password
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        # Insert user
        result = admin.table("users").insert({
            "email": email,
            "password_hash": password_hash
        }).execute()

        user = result.data[0]
        return {"success": True, "user": user}

    except Exception as e:
        print(f"Register error: {e}")
        return {"success": False, "error": str(e)}


def login_user(email, password):
    """Login by checking hashed password directly from database"""
    try:
        admin = get_admin_client()

        # Find user by email
        result = admin.table("users") \
                      .select("*") \
                      .eq("email", email) \
                      .execute()

        if not result.data:
            return {"success": False, "error": "Invalid email or password"}

        user = result.data[0]

        # Check password
        if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            return {"success": True, "user": user}
        else:
            return {"success": False, "error": "Invalid email or password"}

    except Exception as e:
        # Log the real error for debugging (visible in Render logs)
        # but show a generic message to the user so internal details
        # are never exposed in the browser
        print(f"Login error: {e}")
        return {"success": False, "error": "Login failed. Please try again."}


# ── Company profile operations ────────────────────────────────
def get_company_profile(user_id):
    try:
        admin = get_admin_client()
        result = admin.table("company_profiles") \
                      .select("*") \
                      .eq("user_id", user_id) \
                      .execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return None


def save_company_profile(user_id, data):
    try:
        admin = get_admin_client()

        existing = admin.table("company_profiles") \
                        .select("id") \
                        .eq("user_id", user_id) \
                        .execute()

        profile_data = {
            "user_id": user_id,
            "company_name": data.get("company_name", ""),
            "registration_number": data.get("registration_number", ""),
            "pan_number": data.get("pan_number", ""),
            "turnover": float(data.get("turnover", 0) or 0),
            "experience": int(data.get("experience", 0) or 0),
            "domain": data.get("domain", ""),
            "sub_domains": data.get("sub_domains", []),
            "employee_count": int(data.get("employee_count", 0) or 0),
            "certifications": data.get("certifications", ""),
            "address": data.get("address", ""),
            "phone": data.get("phone", ""),
            "company_email": data.get("company_email", ""),
        }

        if existing.data:
            admin.table("company_profiles") \
                 .update(profile_data) \
                 .eq("user_id", user_id) \
                 .execute()
        else:
            admin.table("company_profiles") \
                 .insert(profile_data) \
                 .execute()

        return {"success": True}
    except Exception as e:
        print(f"Error saving profile: {e}")
        return {"success": False, "error": str(e)}


# ── Tender history operations ─────────────────────────────────
def save_tender_analysis(user_id, data):
    try:
        admin = get_admin_client()
        record = {
            "user_id": user_id,
            "project_name": data.get("project_name", "Unknown Project"),
            "project_value": float(data.get("project_value", 0) or 0),
            "location": data.get("location", ""),
            "deadline": data.get("deadline", ""),
            "required_turnover": float(data.get("required_turnover", 0) or 0),
            "required_experience": int(data.get("required_experience", 0) or 0),
            "eligibility_score": int(data.get("eligibility_score", 0) or 0),
            "difficulty_score": int(data.get("difficulty_score", 5) or 5),
            "summary": data.get("summary", ""),
            "recommendations": data.get("recommendations", []),
            "tender_type": data.get("tender_type", ""),
            "overall_eligibility": data.get("overall_eligibility", ""),
            "bid_recommendation": data.get("bid_recommendation", ""),
            "eligibility_criteria": data.get("eligibility_criteria", []),
            "documents_required": data.get("documents_required", []),
            "red_flags": data.get("red_flags", []),
        }
        admin.table("tender_history").insert(record).execute()
        return {"success": True}
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return {"success": False, "error": str(e)}


def get_tender_history(user_id):
    try:
        admin = get_admin_client()
        result = admin.table("tender_history") \
                      .select("*") \
                      .eq("user_id", user_id) \
                      .order("created_at", desc=True) \
                      .execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []


def get_dashboard_stats(user_id):
    try:
        history = get_tender_history(user_id)
        total_analyzed = len(history)
        avg_score = 0
        last_tender = None

        if history:
            scores = [h["eligibility_score"] for h in history if h.get("eligibility_score")]
            avg_score = round(sum(scores) / len(scores)) if scores else 0
            last_tender = history[0]

        return {
            "total_analyzed": total_analyzed,
            "avg_score": avg_score,
            "last_tender": last_tender,
            "recent_history": history[:5]
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "total_analyzed": 0,
            "avg_score": 0,
            "last_tender": None,
            "recent_history": []
        }