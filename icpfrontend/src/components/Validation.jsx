/**
 * @author Louis Figes
 * Basic validations functions for user registration
 */
export const validateName = (name) => {
    if (name.length < 3) {
        return false;
    }
    return true;
}
  
export const validateEmail = (email ) => {
    if (!email.length > 4 && !email.includes('@') || !email.includes('.')) {
        return false;
    }
    return true;
}

export const validatePassword = (password) => {
    if (password.length < 7) {
        return false;
    }
    return true;
}

export const validatePasswordConfirmation = (password, password_confirmation) => {
    if(password === password_confirmation && validatePassword(password)) {
        return true;
    } else {
        return false;
    }
}
