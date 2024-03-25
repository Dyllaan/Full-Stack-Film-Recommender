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
        <HashLoader
            color={"#123abc"}
            loading={loading}
            size={50}
            aria-label="Loading Spinner"
            data-testid="loader"
        />
    );
}

Loading.propTypes = {
    loading: PropTypes.bool
}

export default Loading;