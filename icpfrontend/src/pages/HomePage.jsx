import PreviewFilms from "../components/films/PreviewFilms"
import useAuth from "../components/auth/useAuth";
import GetStartedPage from "./GetStartedPage";
import AuthPage from "./AuthPage";
/**
 * 
 * Simple homepage
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function HomePage() {

  const { signedIn, getStarted } = useAuth();

  return (
    <div className="flex flex-col text-center">
      {signedIn && !getStarted && <PreviewFilms />}
      {getStarted && <GetStartedPage />}
      {!signedIn && 
        <div>
          <h1>Welcome to the Full Stack Film Recommender</h1>
          <p>Sign in or register to get started</p>
          <AuthPage />
        </div>
        }
    </div>
  );
}

export default HomePage;
