import { createContext, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import * as api from './api'; 
import { toast } from 'react-toastify';
import Loading from '../Loading';
import { useNavigate } from 'react-router-dom';

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
  const [refreshToken, setRefreshToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const [requestLoading, setRequestLoading] = useState(false);
  const [getStarted, setGetStarted] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    checkToken();
  }, [loading]);

  if(loading) {
    return <Loading loading={loading} />
  }

  async function getRatings() {
    setRequestLoading(true);
    const response = await api.get('ratings', { Authorization: `Bearer ${accessToken}` });
    if(response.success) {
      setRequestLoading(false);
      if(response.data.results.length === 0) {
        setGetStarted(true);
      }
      setRequestLoading(false);
      return response.data;
    } else {
      setRequestLoading(false);
      toast.error("Your ratings could not be retrieved.");
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
    setLoading(true);
    const response = await api.get('user', { Authorization: `Bearer ${token}` });
    if(response.success) {
      setUserFromCheckCurrent(response.data);
      setAccessToken(token);
      setLoading(false);
    } else {
      await tryToRefreshToken();
      setLoading(false);
    }
  }

  async function tryToRefreshToken() {
    const refreshData = new FormData();
    refreshData.append('refresh', refreshToken);
    const response = await api.post('user/refresh', refreshData);
    if(response.success) {
      setAccessToken(response.data.access);
    } else {
      toast.error("We could not refresh your session, please login again.");
      navigate('/login');
    }
  }

  const login = async(username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const response = await api.post('user/login', formData);
    if (response.success) {
      setUserFromLogin(response.data);
    } else {
      toast.error("Sorry, we could not log you in. Please check your details and try again.");
    }
  }
  
  function setUserFromLogin(response) {
    console.log(response);
    localStorage.setItem('token', response.access);
    localStorage.setItem('refresh', response.refresh);
    setAccessToken(response.access);
    setRefreshToken(response.refresh);
    setUser(response.user);
    setSignedIn(true);
    setLoading(false);
    toast.success("You have successfully logged in.");
  }
  
  function setUserFromCheckCurrent(response) {
    setUser(response);
    setSignedIn(true);
    setLoading(false);
  }

  /**
   * use formdata
   */
  const register = async(values) => {
    const { firstName, lastName, username, email, password } = values;
    const formData = new FormData();
    formData.append('first_name', firstName);
    formData.append('last_name', lastName);
    formData.append('username', username);
    formData.append('email', email);
    formData.append('password', password);
    const response = await api.post('user/register', formData);
    if(response.success) {
      setUserFromLogin(response.data);
      toast.success("You have successfully registered.");
    } else {
      toast.error("Sorry, we could not register you. Please retry.");
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
      toast.success("Your profile has been updated.");
      setUserFromCheckCurrent(response.data);
      return response.data.message;
    } else {
      toast.error("Sorry, we could not update your profile. Please retry.");
      return false;
    }
  }

  AuthProvider.propTypes = {
    children: PropTypes.node.isRequired,
  };

  return (
    <AuthContext.Provider value={{ 
      signedIn, 
      accessToken, 
      user, 
      login, 
      signOut, 
      loading, 
      register, 
      editProfile, 
      getRatings,
      getStarted,
    }}>
      {children}
    </AuthContext.Provider>
  );
}