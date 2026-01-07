# backend/auth.py
import hashlib
import secrets
from datetime import datetime, timedelta
import logging

class AuthenticationSystem:
    """Professional authentication system for Harminder's Platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sessions = {}
        self.version = "2.1.0"
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${password_hash}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = hashed_password.split('$')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == stored_hash
        except:
            return False
    
    def create_session(self, user_id: str, role: str = "user") -> str:
        """Create new user session"""
        session_token = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'role': role,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        self.sessions[session_token] = session_data
        self.logger.info(f"Session created for user: {user_id}")
        return session_token
    
    def validate_session(self, session_token: str) -> dict:
        """Validate user session"""
        if session_token not in self.sessions:
            return None
        
        session_data = self.sessions[session_token]
        
        if datetime.now() > session_data['expires_at']:
            # Session expired
            del self.sessions[session_token]
            return None
        
        # Update expiration time
        session_data['expires_at'] = datetime.now() + timedelta(hours=24)
        
        return session_data
    
    def get_user_permissions(self, role: str) -> dict:
        """Get permissions for user role"""
        permissions = {
            'admin': {
                'view_dashboard': True,
                'manage_users': True,
                'run_models': True,
                'export_data': True,
                'configure_system': True
            },
            'analyst': {
                'view_dashboard': True,
                'manage_users': False,
                'run_models': True,
                'export_data': True,
                'configure_system': False
            },
            'viewer': {
                'view_dashboard': True,
                'manage_users': False,
                'run_models': False,
                'export_data': False,
                'configure_system': False
            }
        }
        
        return permissions.get(role, permissions['viewer'])
    
    def logout(self, session_token: str):
        """Invalidate user session"""
        if session_token in self.sessions:
            user_id = self.sessions[session_token]['user_id']
            del self.sessions[session_token]
            self.logger.info(f"Session terminated for user: {user_id}")