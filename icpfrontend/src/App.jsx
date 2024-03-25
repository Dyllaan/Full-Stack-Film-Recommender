import HomePage from './pages/HomePage';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
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
// Import Swiper styles
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';
/**
 * Main file for the website
 * Stores the layout and routes
 * @author Louis Figes
 */
const App = () => {
  return (
    <div className="flex flex-col min-h-screen">
        <AuthProvider>
          <Header />
          <div className="flex-grow overflow-x-hidden m-4">
            <Routes>
              <Route path="/" element={<HomePage />}/>
              <Route path="/film/:id" element={<FilmPage />}/>
              <Route path="/login" element={<UnAuthed><AuthPage/></UnAuthed>}/>
              <Route path='/register' element={<UnAuthed><AuthPage/></UnAuthed>}/>
              <Route path="/profile" element={<Restricted><ProfilePage /></Restricted>}/>
              <Route path="/logout" element={<Restricted><Logout /></Restricted>}/>
              <Route path="*" element={<PageNotFound />}/>
            </Routes>
          </div>
          <Footer />
        </AuthProvider>
    </div>
  );
};

export default App;
