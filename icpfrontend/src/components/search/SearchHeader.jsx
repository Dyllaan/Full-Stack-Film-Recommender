import Search from './Search';
import PropTypes from 'prop-types';

/**
 * 
 * Header for countries page, has search bar
 * @author Louis Figes
 * 
 */

function SearchHeader({handleSearchChange, message}) {

    return (
        <div className="component-header m-2 flex gap-2">
            <p className="">{message}</p>
            <Search handleSearchChange={handleSearchChange} placeHolder="Search" />
        </div>
    );
}

SearchHeader.propTypes = {
    handleSearchChange: PropTypes.func.isRequired,
    message: PropTypes.string
};

export default SearchHeader;