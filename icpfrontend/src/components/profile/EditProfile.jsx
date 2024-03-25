import { useState, useEffect } from 'react';
import {validateName, validateEmail, validatePassword, validatePasswordConfirmation} from '../Validation';
import useAuth from '../auth/useAuth';
/**
 * 
 * Allows user to edit profile
 * shown on profile page
 * @author Louis Figes
 * @generated Github Copilot was used in the creation of this code.
 */
function EditProfile() {
  const { user, editProfile } = useAuth();
  const [ errors, setErrors ] = useState([]);
  const [ name, setName ] = useState(user.name);
  const [ email, setEmail ] = useState(user.email);
  const [ password, setPassword ] = useState('');
  const [ passwordConfirmation, setPasswordConfirmation ] = useState('');
  
  useEffect(() => {
    if (name === user.name && email === user.email && password === '' && passwordConfirmation === '') {
      setErrors([]);
    } else {
      setErrors(getErrors());
    }
  }, [name, email, password, passwordConfirmation, user]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const changes = getAllChanges();
    if (changes === 'No changes made') {
      setErrors(['No changes made']);
    } else {
      
      editProfile(getChangedVariables());
    }
  };

  /**
   * Only submits the changed variables
   */

  const getChangedVariables = () => {
    const changedVariables = {};
  
    if (hasChanged(name, user.name)) {
      changedVariables.name = name;
    }
    if (hasChanged(email, user.email)) {
      changedVariables.email = email;
    }
    if (hasChanged(password, '')) {
      changedVariables.password = password;
    }
  
    return changedVariables;
  };
  

  const isValid = () => {
    const isNameChangedAndValid = validateName(name) && hasChanged(name, user.name);
    const isEmailChangedAndValid = validateEmail(email) && hasChanged(email, user.email);
    const isPasswordChangedAndValid = validatePassword(password) && hasChanged(password, user.password);
    const isPasswordConfirmationChangedAndValid = validatePasswordConfirmation(password, passwordConfirmation) && hasChanged(passwordConfirmation, user.password);
    const isValid = isNameChangedAndValid || isEmailChangedAndValid || (isPasswordChangedAndValid && isPasswordConfirmationChangedAndValid) ;
    return !isValid;
  };

  const getAllChanges = () => {
    const changes = [];
    if (hasChanged(name, user.name)) {
      changes.push(`Name: ${name}`);
    }
    if (hasChanged(email, user.email)) {
      changes.push(`Email: ${email}`);
    }
    if (hasChanged(password, '')) {
      changes.push(`Password changed`);
    }
    if(changes.length === 0) {
      return 'No changes made';
    } else {
      return (
        <div>
          <h4>Changes</h4>
          <ul>
            {changes.map(change => (
              <li key={change}>{change}</li>
            ))}
          </ul>
        </div>
      )
    }
  };
  
  const getErrors = () => {
    const errors = [];
    if (hasChanged(name, user.name) && !validateName(name)) {
      errors.push('Name must be at least 3 characters');
    }
    if (hasChanged(email, user.email) && !validateEmail(email)) {
      errors.push('Email must be at least 4 characters and contain @ and .');
    }
    if (hasChanged(password, '') && !validatePassword(password)) {
      errors.push('Password must be at least 7 characters');
    }
    if (hasChanged(password, '') &&!validatePasswordConfirmation(password, passwordConfirmation)) {
      errors.push('Password confirmation must match password');
    }
    return errors;
  }

  const hasChanged = (formItem, userItem) => {
    return formItem !== (userItem);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    if (name === 'name') {
      setName(value);
    } else if (name === 'email') {
      setEmail(value);
    } else if (name === 'password') {
      setPassword(value);
    } else if (name === 'passwordConfirmation') {
      setPasswordConfirmation(value);
    }
  }

  return (
    <div className="container px-2">
      <h2 className="text-center">Edit Profile</h2>
      <div className="chi-form w-full" autoComplete="off">
        <input
          id="name"
          name="name"
          type="text"
          placeholder="Name"
          value={name}
          onChange={handleInputChange}
          className={validateName(name) ? 'valid' : 'invalid'}
        />
        <input
          id="email"
          name="email"
          type="email"
          placeholder="Email"
          value={email}
          onChange={handleInputChange}
          className={validateEmail(email) ? 'valid' : 'invalid'}
        />
        <input
          id="password"
          name="password"
          type="password"
          placeholder="Password"
          value={password}
          onChange={handleInputChange}
          className={validatePassword(password) ? 'valid' : 'invalid'}
          autoComplete="off"
        />
        <input
          id="passwordConfirmation"
          name="passwordConfirmation"
          type="password"
          placeholder="Password Confirmation"
          value={passwordConfirmation}
          onChange={handleInputChange}
          className={validatePasswordConfirmation(password, passwordConfirmation) ? 'valid' : 'invalid'}
        />
        <button
          type="submit"
          disabled={isValid()}
          onClick={handleSubmit}
        >
          Make Changes
        </button>
        <div>
          {getAllChanges()}
        </div>
        <div>
          {errors && errors.map(error => (
            <p className="text-red-400 font-bold" key={error}>{error}</p>
          ))}
        </div>
      </div>
    </div>
  );
}

export default EditProfile;
