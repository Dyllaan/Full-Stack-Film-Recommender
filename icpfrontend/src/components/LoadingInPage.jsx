import { Box } from '@mui/material';
import { SquareLoader } from 'react-spinners';

export default function LoadingInPage() {
    return (
      <Box className="flex items-center justify-center h-full w-full">
        <SquareLoader loading={true} color={'#687387'} />
      </Box>
    );
}