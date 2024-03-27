import { createContext, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { useNavigate } from 'react-router-dom';
import * as api from './api'; 
import Message from '../Message';

/**
 * 
 * Handles the user context and rehydration of the user
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

export const AuthContext = createContext();

export default function AuthProvider ({children}) {
  const [user, setUser] = useState(null);
  const [signedIn, setSignedIn] = useState(false);
  const [accessToken, setAccessToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const [requestLoading, setRequestLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [getStarted, setGetStarted] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    checkToken();
  }, [loading]);


  function clearError() {
    setError(null);
  }

  function clearSuccess() {
    setSuccess(null);
  }

  function handleError(message) {
    signOut();
    switch(message) {
      case "Token expired":
        setError("Your session has expired. Please sign in again.");
        break;
      case "Invalid token":
        setError("Your session is invalid. Please sign in again.");
        break;
      default:
        setError("An error occurred. Please sign in again.");
        break;
    }
    navigate("/login");
  }

  async function getRatings() {
    setRequestLoading(true);
    const response = await api.get('ratings', { Authorization: `Bearer ${accessToken}` });
    if(response.success) {
      setRequestLoading(false);
      if(response.data.results.length === 0) {
        setGetStarted(true);
      }
      return response.data;
    } else {
      setRequestLoading(false);
      handleError(response.data.message);
    }

  }

  async function checkToken() {
    if(loading && localStorage.getItem('token') !== null && signedIn === false) {
      const token = localStorage.getItem('token');
      await currentUser(token);
    } else {
      setLoading(false);
    }
  }

  async function currentUser(token) {
    const response = await api.get('user', { Authorization: `Bearer ${token}` });
    if(response.success) {
      setUserFromCheckCurrent(response.data);
      setAccessToken(token);
    } else {
      handleError(response.data.message);
    }
  }
  

  const login = async(username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const response = await api.post('user/login', formData);
    if (response.success) {
      setSuccess("You have successfully logged in.");
      setUserFromLogin(response.data);
    } else {
      signOut();
      setError(response.data.message);
    }
  }
  
  function setUserFromLogin(response) {
    console.log(response);
    localStorage.setItem('token', response.access);
    setAccessToken(response.access);
    setUser(response.user);
    setSignedIn(true);
    setLoading(false);
    setError(null);
  }
  
  function setUserFromCheckCurrent(response) {
    setUser(response);
    setSignedIn(true);
    setLoading(false);
    setError(null);
  }

  /**
   * use formdata
   */
  const register = async(name, email, password) => {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('email', email);
    formData.append('password', password);
    const response = await api.post('user/register', formData);
    if(response.success) {
      setUserFromLogin(response.data);
      setSuccess("You have successfully registered.");
    } else {
      setError(response.data.message);
    }
  }

  function signOut() {
    localStorage.removeItem('token');
    setSignedIn(false);
    setUser(null);
    setAccessToken(null);
    setLoading(false);
  }

  const editProfile = async(data) => {
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${accessToken}`
    };
    
    const response = await api.put('current-user', JSON.stringify(data), headers);
    if(response.success) {
      setSuccess("Profile updated successfully.");
      setUserFromCheckCurrent(response.data);
      return response.data.message;
    } else {
      setError(response.data.message);
      return false;
    }
  }

  AuthProvider.propTypes = {
    children: PropTypes.node.isRequired,
  };

  function displayError() {
    if(error) {
      return <Message message={error} clearMessage={clearError} type={'error'} />
    }
  }

  function displaySuccess() {
    if(success) {
      return <Message message={success} clearMessage={clearSuccess} type={'success'} />
    }
  }

  function displayMessages() {
    return (
      <>
        {displayError()}
        {displaySuccess()}
      </>
    );
  }

  return (
    <AuthContext.Provider value={{ signedIn, accessToken, user, login, signOut, loading, register, editProfile, displayMessages, getRatings  }}>
      {children}
    </AuthContext.Provider>
  );
}