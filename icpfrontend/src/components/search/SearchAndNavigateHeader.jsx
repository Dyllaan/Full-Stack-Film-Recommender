import PropTypes from 'prop-types';
import Search from './Search';
import AwardButton from '../content/AwardButton';
/**
 * 
 * Allows for navigation of content
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function SearchAndNavigateHeader(props) {

    const { nextPage, backPage, page, handleSearchChange, awardButton, award, handleAwardChange } = props;

    return (
        <div className="component-header flex flex-col gap-2">
            <p>Search authors, cities, institutions or content</p>
            <p>Click on an item to go to its contet page</p>
            <div className="w-full">
                <Search handleSearchChange={handleSearchChange} placeHolder="Search"/>
            </div>
            <div className="text-center items-center flex gap-1">
            <button
                onClick={() => nextPage()}
            >
                Next page
            </button>
            <button
                onClick={() => backPage()}
            >
                Go back
            </button>
            <span>Page {page}</span>
            </div>
            {awardButton && (<div>
                <AwardButton handleAwardChange={handleAwardChange} award={award} />
            </div>)}
        </div>
    );
}

SearchAndNavigateHeader.propTypes = {
    nextPage: PropTypes.func.isRequired,
    backPage: PropTypes.func.isRequired,
    page: PropTypes.number.isRequired,
    handleSearchChange: PropTypes.func,
    awardButton: PropTypes.bool,
    award: PropTypes.bool,
    handleAwardChange: PropTypes.func
};

export default SearchAndNavigateHeader;