"""Authorization is a configured dictionary lookup, never an ML prediction."""
import config


def get_user_role(user_id):
    return config.USER_ROLES.get(user_id)


def is_command_allowed(user_id, command):
    role = get_user_role(user_id)
    return command in config.ROLE_PERMISSIONS.get(role, set())
