import PropTypes from 'prop-types';
import TextField from '@mui/material/TextField';
/**
 * 
 * Easy reusable search component 
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function Search({handleSearchChange, placeHolder, styleClass = ""}) {
    return (
        <TextField 
            id="search-bar"
            fullWidth
            autoComplete="off" 
            spellCheck={false} 
            onChange={handleSearchChange} 
            className={styleClass}
            label={placeHolder} variant="outlined"
            color="secondary"
        />
    );
}

Search.propTypes = {
    handleSearchChange: PropTypes.func,
    placeHolder: PropTypes.string.isRequired,
    styleClass: PropTypes.string
};

export default Search;