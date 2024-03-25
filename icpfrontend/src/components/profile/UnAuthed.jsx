import useAuth from "../auth/useAuth";
import PropTypes from 'prop-types';
import { Navigate } from 'react-router-dom';
/**
 * 
 * Prevents a user accessing a page if they are logged in
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function UnAuthed({children}) {
    const { signedIn, isLoading } = useAuth();
    if (signedIn !== true && !isLoading) {
        return children;
    } else {
        return <Navigate to='/' />
    }
}

export default UnAuthed;

UnAuthed.propTypes = {
    children: PropTypes.node.isRequired
}