import { useState } from 'react';
import PropTypes from 'prop-types';
/**
 * 
 * Handles a users login
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function SignIn(props) {
    const { login } = props;

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    function handleEmailChange(e) {
        setEmail(e.target.value);
    }

    function handlePasswordChange(e) {
        setPassword(e.target.value);
    }

    function handleSubmit(e) {
        e.preventDefault();
        login(email, password);
    }

    return (
        <div className="w-full">
            <h2>Sign in</h2>
            <form className="chi-form w-full" onSubmit={handleSubmit}>
                <input
                    name="email"
                    type="text" 
                    placeholder='Email' 
                    onChange={handleEmailChange}
                />
                <input
                    name="password"
                    type="password" 
                    placeholder='Password'
                    onChange={handlePasswordChange}
                />
                <button type="submit">
                    Login
                </button>
            </form>
        </div>
    )
}

SignIn.propTypes = {
    login: PropTypes.func.isRequired
}

export default SignIn;