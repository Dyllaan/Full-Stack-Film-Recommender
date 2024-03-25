import { Link } from "react-router-dom";
/**
 * 
 * A page to display when the user tries to access a page that does not exist.
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function PageNotFound() {
    return (
        <div className="page basic-div">
            <h1>404 Page Not Found</h1>
            <Link to="/">Go Home</Link>
        </div>
    );
}

export default PageNotFound;