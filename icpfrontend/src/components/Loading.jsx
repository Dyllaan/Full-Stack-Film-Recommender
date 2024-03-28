import HashLoader from "react-spinners/HashLoader";
import PropTypes from "prop-types";
/**
 * 
 * Shows a loading message to the user.
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function Loading(props) {
    const {loading} = props;
    return (
        <div className="flex flex-col items-center p-4">
            <HashLoader
                color={"#123abc"}
                loading={loading}
                size={128}
                aria-label="Loading Spinner"
                data-testid="loader"
            />
        </div>
    );
}

Loading.propTypes = {
    loading: PropTypes.bool
}

export default Loading;