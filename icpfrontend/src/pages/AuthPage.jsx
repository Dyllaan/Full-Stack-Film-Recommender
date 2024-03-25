import Register from '../components/profile/Register';
import SignIn from '../components/profile/SignIn';
import useAuth from '../components/auth/useAuth';
/**
 * 
 * A login  / register page.
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function AuthPage() {
  const {login, register} = useAuth();

  return (
    <div className="page flex flex-col">
      <div className="w-[80vw] container text-center flex flex-col lg:flex-row mx-auto overflow-hidden">
        <SignIn login={login} /> 
        <Register register={register} />
      </div>
      <div>
        <p className="text-center">This website is <span className="text-red-800 font-bold">NOT</span> an official CHI website or in any way affiliated.</p>
      </div>
    </div>
    );
}

export default AuthPage;