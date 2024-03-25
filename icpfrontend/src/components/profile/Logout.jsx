import useAuth from "../auth/useAuth";
import { useEffect } from "react";
/**
 * 
 * Logs the user out, this allows for the /logout route
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function Logout() {
    const { signOut, signedIn } = useAuth();

    useEffect(() => {
        if(signedIn)
            signOut();
    }, [signedIn]);

    return (
        <div>
            <h1>Signed Out</h1>
        </div>
    );
}

export default Logout;