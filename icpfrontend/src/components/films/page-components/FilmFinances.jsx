import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import PropTypes from 'prop-types';

export default function FilmFinances ({budget, revenue}) {

    const gross = revenue - budget;

    const profitable = gross > 0;

    return (
        <TableContainer component={Paper}>
        <Table aria-label="financial table">
            <TableBody>
            <TableRow>
                <TableCell component="th" scope="row">Budget</TableCell>
                <TableCell align="right">${budget.toLocaleString()}</TableCell>
            </TableRow>
            <TableRow>
                <TableCell component="th" scope="row">Revenue</TableCell>
                <TableCell align="right">${revenue.toLocaleString()}</TableCell>
            </TableRow>
            <TableRow>
                <TableCell component="th" scope="row">Gross</TableCell>
                <TableCell align="right"><span className={profitable ? 'text-green-600' : 'text-red-600'}>${gross.toLocaleString()}</span></TableCell>
            </TableRow>
            </TableBody>
        </Table>
        </TableContainer>
    );
}

FilmFinances.propTypes = {
    budget: PropTypes.number,
    revenue: PropTypes.number
}