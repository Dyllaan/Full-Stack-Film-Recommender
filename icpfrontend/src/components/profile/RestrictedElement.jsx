import {Link} from "react-router-dom";
import PropTypes from 'prop-types';
/**
 * 
 * Prevents unauthenticated users from using an element
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function RestrictedElement(props) {
    const text = props.text;
    return (
        <div className="basic-div">
            <Link to="/login">
                <h2>Sign in to {text}</h2>
            </Link>
        </div>
    )
}

RestrictedElement.propTypes = {
    text: PropTypes.string.isRequired,
};

export default RestrictedElement;