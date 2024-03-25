import PropTypes from 'prop-types';
import { NavLink } from 'react-router-dom';
/**
 * 
 * Nav Item for the header
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function NavItem(props) {
    const { pageName, pageUrl } = props;
    return (
        <NavLink
            className={({ isActive }) =>
              isActive ? "nav-item active" : "nav-item"
            }
            to={pageUrl}
        >
            {pageName}
        </NavLink>
    );
}

NavItem.propTypes = {
    pageName: PropTypes.string.isRequired,
    pageUrl: PropTypes.string.isRequired,
};

export default NavItem;