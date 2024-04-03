import { useState, useEffect } from 'react';
import useFetchData from '../../hooks/useFetchData';
import Search from '../Search';
import { Box, Paper, Popover, Stack, MenuItem, Divider } from '@mui/material';
import { Link } from 'react-router-dom';
/**
 * 
 * Show all the countries from the affiliation table, allows for searching of countries by name
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 * @author nymphadora @ StackOverflow
 * April 7 2021.
 * https://stackoverflow.com/questions/51388205/prevent-auto-focus-of-a-material-ui-popover-element
 * Disable auto focus of popover
 * 
 */

function FilmSearch() {
    const endpoint = 'movies';
    const { data, setEndpoint, reloadData, doRun } = useFetchData(endpoint, false);
    const [debouncedSearch, setDebouncedSearch] = useState("");
    const [search, setSearch] = useState("");
    const [anchorEl, setAnchorEl] = useState(null);
    const open = Boolean(anchorEl);

    const [popoverWidth, setPopoverWidth] = useState(0);

    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedSearch(search);
            doRun();
        }, 300);
        return () => {
            clearTimeout(timer);
        };
    }, [search]);

    useEffect(() => {
        let newEndpoint = `${endpoint}`;
        
        if (debouncedSearch) {
            newEndpoint += `?search=${debouncedSearch}`;
        setEndpoint(newEndpoint);
        reloadData();
        }
    }, [endpoint, debouncedSearch]);

    useEffect(() => {
        const searchWidth = document.getElementById('search-bar').offsetWidth;
        setPopoverWidth(searchWidth);
      }, []);
    
    const handleClose = () => {
        setAnchorEl(null);
    };

    const handleSearchChange = (event) => {
        setSearch(event.target.value);
        setAnchorEl(event.currentTarget);
    };

    const renderedData = data.results ? data.results.slice(0, 10) : [];

    return (
        <div className="flex flex-col w-[50vw] mx-auto">
            <div className="p-2">
                <Search handleSearchChange={handleSearchChange} placeHolder="Search films"/>
            </div>
            <Popover
                open={open}
                anchorEl={anchorEl}
                onClose={handleClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'left',
                }}
                disableAutoFocus={true}
                disableEnforceFocus={true}
            >
                <Paper style={{ width: popoverWidth }}>
                    <Stack>
                        {data && data.results && data.results.length > 0 && renderedData.map((film, index) => (
                            <Box key={index}>
                                <Divider />
                                <Link to={`/film/${film.movie_slug}`}>
                                    <MenuItem underline="hover">
                                        <Box className="p-2">
                                            <p>{film.movie_title}</p>
                                        </Box>
                                    </MenuItem>
                                </Link>
                            </Box>
                        ))}
                    </Stack>
                </Paper>
            </Popover>
        </div>
    );
}

export default FilmSearch;
