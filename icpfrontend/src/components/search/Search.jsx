import PropTypes from 'prop-types';
/**
 * 
 * Easy reusable search component 
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function Search({handleSearchChange, placeHolder}) {
    return (
        <input
            type="text"
            placeholder={placeHolder}
            onChange={handleSearchChange}
            className="search-bar"
        />
    );
}

Search.propTypes = {
    handleSearchChange: PropTypes.func,
    placeHolder: PropTypes.string.isRequired
};

export default Search;