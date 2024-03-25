import { useContext } from 'react';
import { AuthContext } from './AuthProvider';
/**
 * 
 * Allows components to use the AuthContext and access the user object and token
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function useAuth() {
    return useContext(AuthContext);
}

export default useAuth;