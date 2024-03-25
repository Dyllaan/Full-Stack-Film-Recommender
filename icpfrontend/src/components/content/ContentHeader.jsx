import PropTypes from 'prop-types';
import Search from '../search/Search';
import AwardButton from './AwardButton';
/**
 * 
 * Allows for navigation of content
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function ContentHeader(props) {

    const { contentTypes, setType, nextPage, backPage, page, setAward, handleSearchChange, award } = props;


    const handleAwardChange = (e) => {
        setAward(e.target.checked);
    }

    return (
        <div className="flex flex-wrap gap-2 items-center">
            <div className="w-full">
                <Search handleSearchChange={handleSearchChange} placeHolder="Search Content"/>
            </div>
            <select onChange={(e) => setType(e.target.value)} defaultValue="all">
                <option value="all">All</option>
                {contentTypes && contentTypes.map((item, index) => (
                    <option key={index} value={item.type}>{item.type}</option>
                ))}
            </select>
            <button onClick={() => nextPage()}>
                Next page
            </button>
            <button onClick={() => backPage()}>
                Go back
            </button>
            <span className="font-bold text-slate-400">Page {page}</span>
            <div>
                <AwardButton handleAwardChange={handleAwardChange} award={award} />
            </div>
        </div>
    );
}

ContentHeader.propTypes = {
    contentTypes: PropTypes.array.isRequired,
    nextPage: PropTypes.func.isRequired,
    backPage: PropTypes.func.isRequired,
    setType: PropTypes.func.isRequired,
    page: PropTypes.number.isRequired,
    setAward : PropTypes.func,
    handleSearchChange: PropTypes.func,
    award: PropTypes.bool
};

export default ContentHeader;