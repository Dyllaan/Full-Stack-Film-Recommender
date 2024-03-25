import NavItem from './NavItem';
import { NavLink } from 'react-router-dom';
import useAuth from '../auth/useAuth';
import UserDropdown from './UserDropdown';
/**
 * 
 * Header & navbar
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
function Header() {
    const { signedIn, user, displayMessages } = useAuth();

    function showUserNavItems() {
        if (signedIn) {
            return (
                <UserDropdown user={user} />
            );
        } else {
            return (
                <>
                    <NavItem pageName="Login" pageUrl="/login" />
                    <NavItem pageName="Register" pageUrl="/register" />
                </>
            );
        }
    }

    return (
        <header>
            <nav className="flex items-center justify-between py-2 px-8 w-screen">
                <div className="flex items-center">
                    <NavLink to="/"><h1>CHI 2023</h1></NavLink>
                    <NavItem pageName="Content" pageUrl="/content" />
                    <NavItem pageName="Countries" pageUrl="/countries" />
                </div>
                <div className="flex items-center">
                    {displayMessages()}
                    {showUserNavItems()}
                </div>
            </nav>
        </header>
    );
}

export default Header;