import { useState } from "react";
import { validateEmail, validateName, validatePassword, validatePasswordConfirmation } from "../Validation";
import PropTypes from 'prop-types';
/**
 * Shows registration form and handles registration and some validation
 * @author Louis Figes
 * @generated Github Copilot assisted with this code 
 */
function Register(props) {
  const {register} = props;

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    password_confirmation: ""
  });

  const [touched, setTouched] = useState({
    name: false,
    email: false,
    password: false,
    password_confirmation: false
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    setTouched({
      ...touched,
      [name]: true
    });
  };

  function handleSubmit(e) {
    e.preventDefault();
    register(formData.name, formData.email, formData.password, formData.password_confirmation)
  }

  const isValid = () => {
    return !validateName(formData.name) || !validateEmail(formData.email) || !validatePassword(formData.password) || !validatePasswordConfirmation(formData.password, formData.password_confirmation);
  }

  return (
    <div className="w-full">
      <h2>Register</h2>
      <form className="chi-form w-full" onSubmit={handleSubmit}>
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleInputChange}
          className={validateName(formData.name) ? 'valid' : 'invalid'}
          placeholder="Name"
        />
        <input
          type="text"
          name="email"
          value={formData.email}
          onChange={handleInputChange}
          className={validateEmail(formData.email) ? 'valid' : 'invalid'}
          placeholder="Email"
        />
        <input
          type="password"
          name="password"
          value={formData.password}
          onChange={handleInputChange}
          className={validatePassword(formData.password) ? 'valid' : 'invalid'}
          placeholder="Password"
        />
        <input
          type="password"
          name="password_confirmation"
          value={formData.password_confirmation}
          onChange={handleInputChange}
          className={validatePasswordConfirmation(formData.password, formData.password_confirmation) ? 'valid' : 'invalid'}
          placeholder="Confirm Password"
        />
        <button
          type="submit"
          disabled={isValid()}
        >
          Register
        </button>
        <div className="error-messages">
          {!validateName(formData.name) && touched.name && <p className="error-message">Name must be at least 3 characters</p>}
          {!validateEmail(formData.email) && touched.email && <p className="error-message">Email is invalid</p>}
          {!validatePassword(formData.password) && touched.password && <p className="error-message">Password must be at least 8 characters</p>}
          {!validatePasswordConfirmation(formData.password, formData.password_confirmation) && touched.password_confirmation && <p className="error-message">Passwords must match</p>}
        </div>
      </form>
    </div>
  );
}

Register.propTypes = {
  register: PropTypes.func.isRequired
}

export default Register;
