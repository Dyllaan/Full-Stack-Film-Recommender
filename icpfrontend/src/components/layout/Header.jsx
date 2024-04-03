import { useState } from 'react';
import { AppBar, Toolbar, Box, Typography, Menu, Tooltip, MenuItem, Grid } from '@mui/material';
import useAuth from '../auth/useAuth';
import MovieIcon from '@mui/icons-material/Movie';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';
import FilmSearch from '../films/FilmSearch';
import ThemeToggle from './themes/ThemeToggle';

function Header({toggleTheme, theme}) {
  const [anchorUser, setAnchorUser] = useState(null);
  const { signedIn, user } = useAuth();

  const handleOpenUserMenu = (event) => {
    setAnchorUser(event.currentTarget);
  };

  const handleCloseUserMenu = () => {
    setAnchorUser(null);
  };

  return (
    <AppBar position="sticky" className="py-2 px-8 w-full">
      <Toolbar disableGutters className="relative items-center justify-space-between">
        <Grid container className="items-center justify-space-between">
          {/* Left Section */}
          <Grid item xs>
            <Grid container className="items-center gap-1">
              <Link to="/" className="flex items-center gap-1">
                    <MovieIcon />
                    <Typography variant="h6" sx={{
                          fontWeight: 500,   
                          color: 'inherit',
                          textDecoration: 'none',
                          }}>
                              ICP
                      </Typography>
                      <Box className="flex flex-col ml-2">
                        <Typography className="text-gray-200" variant="subtitle2">
                          Powered by
                        </Typography>
                        <Typography className="text-yellow-300" 
                        sx={{
                        fontWeight: 700,
                        }}
                        variant="caption">
                          TMDb
                        </Typography>
                        <Typography className="text-red-300" variant="caption"
                        sx={{
                          fontWeight: 700,
                          }}>
                          MovieLens
                        </Typography>
                      </Box>
                  </Link>
                </Grid>
            </Grid>

            {/* Middle Section */}
            <Grid item xs display="flex" justifyContent="center">
              <FilmSearch />
            </Grid>

            {/* Right Section */}
            <Grid item xs display="flex" justifyContent="flex-end" alignItems="center">
              <ThemeToggle toggleTheme={toggleTheme} theme={theme} />
            {signedIn && (
                <>
                    <Tooltip title="You">
                        <MenuItem variant="outlined" onClick={handleOpenUserMenu} sx={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                            <Typography variant="h6" component="button" href="/" sx={{
                                fontWeight: 300,
                                color: 'inherit',
                                textDecoration: 'none',
                                
                            }}>
                                {user.username}
                            </Typography>
                            <svg className={`-mr-1 ml-2 h-5 w-5 transform transition-transform ${anchorUser ? 'rotate-180' : ''}`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="white" aria-hidden="true">
                                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 011.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                </svg>
                        </MenuItem>
                    </Tooltip>
                    <Menu
                        sx={{ mt: '45px' }}
                        anchorEl={anchorUser}
                        anchorOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                        }}
                        keepMounted
                        transformOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                        }}
                        open={Boolean(anchorUser)}
                        onClose={handleCloseUserMenu}
                    >
                        <MenuItem onClick={handleCloseUserMenu}>
                            <Typography textAlign="center"><Link to="/profile">Profile</Link></Typography>
                        </MenuItem>
                        <MenuItem onClick={handleCloseUserMenu}>
                            <Typography textAlign="center"><Link to="/logout">Logout</Link></Typography>
                        </MenuItem>
                    </Menu>
                </>
            )}
        </Grid>
        </Grid>
        </Toolbar>
    </AppBar>
  );
}

Header.propTypes = {
  toggleTheme: PropTypes.func.isRequired,
  theme: PropTypes.string.isRequired,
}

export default Header;
