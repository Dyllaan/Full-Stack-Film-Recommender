import HomePage from './pages/HomePage';
import Header from './components/layout/Header';
import { Routes, Route } from 'react-router-dom';
import './App.css'
import AuthPage from './pages/AuthPage';
import AuthProvider from './components/auth/AuthProvider';
import ProfilePage from './pages/ProfilePage';
import Restricted from './components/profile/Restricted';
import Logout from './components/profile/Logout';
import UnAuthed from './components/profile/UnAuthed';
import PageNotFound from './pages/PageNotFound';
import FilmPage from './pages/FilmPage';
import GetStartedPage from './pages/GetStartedPage';
// Import Swiper styles
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useState, useEffect } from 'react';
import ThemeWrapper from './components/layout/themes/ThemeWrapper';
import CssBaseline from '@mui/material/CssBaseline';
/**
 * Main file for the website
 * Stores the layout and routes
 * @author Louis Figes
 */

const App = () => {

  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    const storedTheme = localStorage.getItem('theme') || 'light';
    setTheme(storedTheme);
  }, []);

  useEffect(() => {
    localStorage.setItem('theme', theme);
  }, [theme]);

  return (
    <div className="flex flex-col min-h-screen">
        <AuthProvider>
        <ThemeWrapper theme={theme}>
          <CssBaseline />
          <Header toggleTheme={toggleTheme} theme={theme}/>
          <div className="flex-grow overflow-x-hidden m-4">
            <Routes>
              <Route path="/" element={<HomePage />}/>
              <Route path="/film/:slug" element={<FilmPage />}/>
              <Route path="/login" element={<UnAuthed><AuthPage/></UnAuthed>}/>
              <Route path='/register' element={<UnAuthed><AuthPage/></UnAuthed>}/>
              <Route path="/profile" element={<Restricted><ProfilePage /></Restricted>}/>
              <Route path="/logout" element={<Restricted><Logout /></Restricted>}/>
              <Route path="/get-started" element={<Restricted><GetStartedPage /></Restricted>}/>
              <Route path="*" element={<PageNotFound />}/>
            </Routes>
          </div>
          <ToastContainer />
          </ThemeWrapper>
        </AuthProvider>
    </div>
  );
};

export default App;
