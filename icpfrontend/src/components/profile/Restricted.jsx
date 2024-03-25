import useAuth from "../auth/useAuth";
import PropTypes from 'prop-types';
import { Navigate } from 'react-router-dom';
/**
 * 
 * Prevents logged in users from accessing a page
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function Restricted({children}) {
    const { signedIn, isLoading } = useAuth();
    if (signedIn && !isLoading) {
        return children;
    } else {
        return <Navigate to='/' />
    }
}

export default Restricted;

Restricted.propTypes = {
    children: PropTypes.node.isRequired
}