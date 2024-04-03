import Register from '../components/profile/Register';
import SignIn from '../components/profile/SignIn';
import useAuth from '../components/auth/useAuth';
import { useState } from 'react';
import { Button, Box } from '@mui/material';
/**
 * 
 * A login  / register page.
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function AuthPage() {
  const {login, register} = useAuth();
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="flex flex-col">
      <div className="w-[80vw] container text-center flex flex-col lg:flex-row mx-auto overflow-hidden">
        {isLogin ? <SignIn login={login} /> : <Register register={register} />}
        <Box>
          <Button onClick={() => setIsLogin(!isLogin)}>{isLogin ? "Register" : "Login"}</Button>
        </Box>
      </div>
      <div>
        <p className="text-center text-lg">This website is based on the <span className="text-red-800 font-bold">MovieLens</span> dataset</p>
      </div>
    </div>
    );
}

export default AuthPage;