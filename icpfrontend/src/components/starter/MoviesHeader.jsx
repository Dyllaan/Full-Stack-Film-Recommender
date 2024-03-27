import PropTypes from 'prop-types';
import Search from '../search/Search';
/**
 * 
 * Allows for navigation of content
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function MoviesHeader(props) {
    const { handleSearchChange } = props;
    return (
        <div className="flex flex-wrap gap-2 items-center">
            <Search handleSearchChange={handleSearchChange} placeHolder="Search Content"/>
        </div>
    );
}

MoviesHeader.propTypes = {
    handleSearchChange: PropTypes.func,
};

export default MoviesHeader;